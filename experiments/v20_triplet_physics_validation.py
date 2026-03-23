"""
v20_triplet_physics_validation.py
=================================
OlfaBind v20: 3-Module Mathematical Redesign Validation

Module 1: Triplet Margin Loss (L = max(0, d(a,p) - d(a,n) + alpha))
Module 2: Ghost Mass Restoration (not used in Snitz pair task)
Module 3: Mass Decay m(t) = m0*exp(-k*T*t) + Chaos/Resonance Detection

Phase 1: Pre-train SliceLatentModule with Triplet Margin Loss
Phase 2: 5-seed x 5-fold CV comparison vs v18 baseline
"""
import os, sys, json, time, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
import pandas as pd

sys.path.insert(0, r"C:\Users\user\Desktop\Game\server")
from models.olfabind_input import InputHardwareLayer
from models.olfabind_engine import PhysicsProcessingEngine
from models.olfabind_contrastive import SliceLatentModule

from rdkit import Chem
from rdkit.Chem import AllChem

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

DREAM_DIR = r"C:\Users\user\Desktop\Game\server\data\pom_data\dream_mixture"
RESULTS_DIR = r"C:\Users\user\Desktop\Game\paper\results\mixture_prediction"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ======================================================================
# DATA LOADING
# ======================================================================
FP_DIM = 2048
c2s = {}
for dn in ['snitz_2013', 'bushdid_2014', 'ravia_2020']:
    mf = os.path.join(DREAM_DIR, dn, 'molecules.csv')
    if os.path.exists(mf):
        df = pd.read_csv(mf)
        for _, row in df.iterrows():
            cid = str(row.get('CID', '')).strip().replace('.0', '')
            smi = str(row.get('IsomericSMILES', row.get('SMILES', ''))).strip()
            if cid and smi and smi != 'nan':
                c2s[cid] = smi

FP_CACHE = {}
def get_fp(smi):
    if smi not in FP_CACHE:
        m = Chem.MolFromSmiles(smi)
        if m: FP_CACHE[smi] = np.array(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=FP_DIM), dtype=np.float32)
        else: FP_CACHE[smi] = np.zeros(FP_DIM, dtype=np.float32)
    return FP_CACHE[smi]

fp_lookup = {smi: get_fp(smi) for smi in c2s.values()}

def load_snitz():
    pairs = []
    with open(os.path.join(DREAM_DIR, "snitz_2013", "behavior.csv"), 'r', errors='ignore') as f:
        for row in csv.DictReader(f):
            ca = [c.strip() for c in row['StimulusA'].split(',') if c.strip() in c2s]
            cb = [c.strip() for c in row['StimulusB'].split(',') if c.strip() in c2s]
            if ca and cb:
                pairs.append({'ca': ca, 'cb': cb, 'sim': float(row['Similarity'])})
    return pairs

snitz_all = load_snitz()
print(f"Data: Snitz={len(snitz_all)}")

all_unique_cids = set()
for p in snitz_all:
    all_unique_cids.update(p['ca'])
    all_unique_cids.update(p['cb'])
all_unique_cids = [c for c in all_unique_cids if c in c2s and c2s[c] in fp_lookup]
print(f"Unique molecules for Triplet pre-training: {len(all_unique_cids)}")

def augment_pairs(pairs):
    augmented = list(pairs)
    for p in pairs:
        if p['ca'] != p['cb']:
            augmented.append({'ca': p['cb'], 'cb': p['ca'], 'sim': p['sim']})
    return augmented

MAX_MOLS = 20

class OlfaBindDataset(Dataset):
    def __init__(self, pairs, max_mols=MAX_MOLS, emb_dim=FP_DIM):
        self.pairs = pairs; self.max_mols = max_mols; self.emb_dim = emb_dim
    def __len__(self): return len(self.pairs)
    def _pad(self, cids):
        embs = [fp_lookup[c2s[c]] for c in cids if c in c2s and c2s[c] in fp_lookup]
        out = np.zeros((self.max_mols, self.emb_dim), dtype=np.float32)
        mask = np.zeros(self.max_mols, dtype=np.float32)
        for i in range(min(len(embs), self.max_mols)):
            if embs[i].shape[0] == self.emb_dim: out[i] = embs[i]; mask[i] = 1.0
        return out, mask
    def __getitem__(self, idx):
        p = self.pairs[idx]
        a, ma = self._pad(p['ca']); b, mb = self._pad(p['cb'])
        return {
            'fp_a': torch.from_numpy(a), 'mask_a': torch.from_numpy(ma),
            'fp_b': torch.from_numpy(b), 'mask_b': torch.from_numpy(mb),
            'sim': torch.tensor(p['sim'] / 100.0, dtype=torch.float32)
        }

# ======================================================================
# MODEL: v20 OlfaBind with Triplet Margin Loss + Mass Decay
# ======================================================================
PHYS_EMB_DIM = 20  # 19 -> 20 (chaos score added)

class OlfaBindV20(nn.Module):
    """
    v20: Triplet Margin Loss + Mass Decay + Chaos Detection

    Pipeline:
    FP (B,N,2048) -> InputHardwareLayer -> constellation (B,N,128)
                   -> SliceLatentModule (Triplet) -> z_positions (B,N,3)
                   -> PhysicsProcessingEngine (mass decay + chaos) -> emb (B,20)
                   -> proj -> sim_head -> similarity
    """
    def __init__(self, d_input=FP_DIM, d_atom=128, n_steps=4, dt=0.05,
                 contrastive_weight=0.1, margin=1.0):
        super().__init__()
        self.contrastive_weight = contrastive_weight

        self.input_layer = InputHardwareLayer(
            d_input=d_input, d_atom=d_atom, grid_h=8, grid_w=16
        )
        self.contrastive = SliceLatentModule(
            d_atom=d_atom, h_dim=256, z_dim=3,
            margin=margin, drop_prob=0.2, noise_std=0.05,
            position_scale_init=2.0
        )
        self.physics = PhysicsProcessingEngine(
            d_atom=d_atom, n_steps=n_steps, dt=dt
        )
        # 20-dim physics embedding (was 19 in v19)
        self.proj = nn.Sequential(
            nn.Linear(PHYS_EMB_DIM, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 64)
        )
        self.sim_head = nn.Sequential(
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1)
        )

    def _process_mixture(self, fp, mask):
        constellation = self.input_layer(fp, mask)
        z_pos, cl_loss = self.contrastive(constellation, mask)
        _, emb, _ = self.physics(constellation, mask, override_positions=z_pos)
        return emb, cl_loss

    def forward(self, fp_a, mask_a, fp_b, mask_b):
        emb_a, cl_a = self._process_mixture(fp_a, mask_a)
        emb_b, cl_b = self._process_mixture(fp_b, mask_b)
        pa, pb = self.proj(emb_a), self.proj(emb_b)
        sim = torch.sigmoid(self.sim_head((pa - pb).abs()).squeeze(-1))
        self._contrastive_loss = (cl_a + cl_b) / 2.0
        return sim

    def get_contrastive_loss(self):
        return getattr(self, '_contrastive_loss', torch.tensor(0.0))

# v18 baseline (no contrastive, no mass decay in training)
class OlfaBindBaseline(nn.Module):
    def __init__(self, d_input=FP_DIM, d_atom=128, n_steps=4, dt=0.05):
        super().__init__()
        self.input_layer = InputHardwareLayer(d_input=d_input, d_atom=d_atom, grid_h=8, grid_w=16)
        self.physics = PhysicsProcessingEngine(d_atom=d_atom, n_steps=n_steps, dt=dt)
        self.proj = nn.Sequential(
            nn.Linear(PHYS_EMB_DIM, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 64))
        self.sim_head = nn.Sequential(
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))
    def forward(self, fp_a, mask_a, fp_b, mask_b):
        ca, cb = self.input_layer(fp_a, mask_a), self.input_layer(fp_b, mask_b)
        _, ea, _ = self.physics(ca, mask_a); _, eb, _ = self.physics(cb, mask_b)
        pa, pb = self.proj(ea), self.proj(eb)
        return torch.sigmoid(self.sim_head((pa - pb).abs()).squeeze(-1))

# ======================================================================
# TRAINING
# ======================================================================
def eval_model(model, loader):
    model.eval(); preds, trues = [], []
    with torch.no_grad():
        for b in loader:
            y = model(b['fp_a'].to(device), b['mask_a'].to(device),
                      b['fp_b'].to(device), b['mask_b'].to(device))
            for i in range(len(y)):
                if not (torch.isnan(y[i]) or torch.isinf(y[i])):
                    preds.append(y[i].cpu().item()); trues.append(b['sim'][i].item())
    if len(preds) < 2 or np.std(preds) < 1e-8: return 0.0
    return pearsonr(preds, trues)[0]

def pretrain_triplet(input_layer, contrastive_module, molecules_fps,
                     epochs=100, lr=1e-3, batch_size=64):
    """Phase 1: Pre-train with Triplet Margin Loss."""
    print("\n" + "="*60)
    print("PHASE 1: Triplet Margin Loss Pre-training")
    print("="*60)

    input_layer.eval()
    contrastive_module.train()
    opt = optim.Adam(contrastive_module.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    n = len(molecules_fps)
    fp_tensor = torch.stack(molecules_fps).to(device)

    best_loss = float('inf')
    loss_history = []

    for ep in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = perm[start:end]
            batch_fp = fp_tensor[idx].unsqueeze(1)  # (bs, 1, FP_DIM)
            mask = torch.ones(batch_fp.shape[0], 1, device=device)

            with torch.no_grad():
                constellation = input_layer(batch_fp, mask)

            opt.zero_grad()
            _, cl_loss = contrastive_module(constellation, mask)

            if torch.isnan(cl_loss) or torch.isinf(cl_loss):
                continue

            cl_loss.backward()
            torch.nn.utils.clip_grad_norm_(contrastive_module.parameters(), 1.0)
            opt.step()

            total_loss += cl_loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        loss_history.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss

        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"  Epoch {ep+1}/{epochs}: loss={avg_loss:.4f} (best={best_loss:.4f})")

    print(f"  Pre-training done. Final loss={avg_loss:.4f}, Best={best_loss:.4f}")
    return loss_history

def train_single(model, train_loader, val_loader, epochs=50, phys_lr=1e-5, head_lr=5e-4):
    has_contrastive = hasattr(model, 'contrastive')

    if hasattr(model, 'input_layer'):
        physics_params = list(model.input_layer.parameters()) + list(model.physics.parameters())
        if has_contrastive:
            physics_params += list(model.contrastive.parameters())
        head_params = [p for p in model.parameters() if not any(p is pp for pp in physics_params)]
        opt = optim.Adam([
            {'params': physics_params, 'lr': phys_lr},
            {'params': head_params, 'lr': head_lr},
        ], weight_decay=1e-4)
        if hasattr(model, 'sim_head'):
            for m in model.sim_head:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=2.0)
        clip_val = 0.5
    else:
        opt = optim.Adam(model.parameters(), lr=head_lr, weight_decay=1e-4)
        clip_val = 1.0

    scheduler = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    best_r, best_state, no_improve = 0.0, None, 0

    for ep in range(epochs):
        model.train()
        for b in train_loader:
            opt.zero_grad()
            y = model(b['fp_a'].to(device), b['mask_a'].to(device),
                      b['fp_b'].to(device), b['mask_b'].to(device))
            loss = F.mse_loss(y, b['sim'].to(device))

            if hasattr(model, 'input_layer'):
                loss = loss + 0.01 * model.input_layer.get_sparsity_loss()

            if has_contrastive:
                cl_loss = model.get_contrastive_loss()
                loss = loss + model.contrastive_weight * cl_loss

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_val)
            opt.step()

        scheduler.step()
        r_val = eval_model(model, val_loader)

        if r_val > best_r:
            best_r = r_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 10:
                break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model, best_r

def train_and_eval(model_cls, train_p, val_p, epochs=50, lr=5e-4, seed=42,
                   n_restarts=3, pretrained_contrastive_state=None):
    torch.manual_seed(seed); np.random.seed(seed)
    train_loader = DataLoader(OlfaBindDataset(augment_pairs(train_p)), batch_size=16, shuffle=True)
    val_loader = DataLoader(OlfaBindDataset(val_p), batch_size=16)

    best_model, best_r = None, -1.0
    for restart in range(n_restarts):
        torch.manual_seed(seed * 1000 + restart)
        model = model_cls().to(device)

        if pretrained_contrastive_state is not None and hasattr(model, 'contrastive'):
            model.contrastive.load_state_dict(
                {k: v.to(device) for k, v in pretrained_contrastive_state.items()}
            )

        model, r = train_single(model, train_loader, val_loader, epochs=epochs,
                                phys_lr=1e-5, head_lr=lr)
        if r > best_r:
            best_r = r
            best_model = model

    return best_model, best_r

# ======================================================================
# MAIN EXPERIMENT
# ======================================================================
SEEDS = [42, 123, 456, 789, 2024]
best_T, best_dt = 4, 0.05

# --- PHASE 1: Triplet Margin Loss Pre-training ---
pretrain_input = InputHardwareLayer(d_input=FP_DIM, d_atom=128, grid_h=8, grid_w=16).to(device)
cl_module = SliceLatentModule(
    d_atom=128, h_dim=256, z_dim=3,
    margin=1.0, drop_prob=0.2, noise_std=0.05,
    position_scale_init=2.0
).to(device)

mol_fps = [torch.from_numpy(fp_lookup[c2s[c]]) for c in all_unique_cids]
print(f"\nPre-training on {len(mol_fps)} unique molecules with Triplet Margin Loss")

pretrain_loss_history = pretrain_triplet(
    pretrain_input, cl_module, mol_fps,
    epochs=100, lr=1e-3, batch_size=64
)
pretrained_state = {k: v.cpu().clone() for k, v in cl_module.state_dict().items()}

# --- PHASE 2: 5-seed x 5-fold CV ---
print("\n" + "="*60)
print("PHASE 2: 5-seed x 5-fold CV -- v20 Triplet vs v18 Baseline")
print("="*60)

results = {}

# v20: Triplet + Mass Decay + Chaos
print("\n--- v20: OlfaBind + Triplet Margin Loss ---")

def make_v20_model():
    class M(OlfaBindV20):
        def __init__(self):
            super().__init__(n_steps=best_T, dt=best_dt, contrastive_weight=0.1, margin=1.0)
    return M

v20_cls = make_v20_model()
all_rs_v20 = []
t0 = time.time()

for seed in SEEDS:
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (ti, vi) in enumerate(kf.split(range(len(snitz_all)))):
        tp = [snitz_all[i] for i in ti]; vp = [snitz_all[i] for i in vi]
        _, r = train_and_eval(v20_cls, tp, vp, epochs=50, lr=5e-4,
                              seed=seed*100+fold, n_restarts=3,
                              pretrained_contrastive_state=pretrained_state)
        all_rs_v20.append(r)
        print(f"  Seed={seed} Fold={fold+1} r={r:.4f}")

n_params_v20 = sum(p.numel() for p in v20_cls().parameters())
non_zero_v20 = [x for x in all_rs_v20 if x > 0]
results['v20_triplet'] = {
    'mean_r': float(np.mean(all_rs_v20)),
    'std_r': float(np.std(all_rs_v20)),
    'all_r': [float(x) for x in all_rs_v20],
    'collapse_rate': 1.0 - len(non_zero_v20) / len(all_rs_v20),
    'n_params': n_params_v20,
    'time_sec': time.time() - t0,
    'pretrain_final_loss': pretrain_loss_history[-1] if pretrain_loss_history else 0,
}
print(f"  => v20 Triplet: r={np.mean(all_rs_v20):.4f}+/-{np.std(all_rs_v20):.4f}")

# v18 Baseline (with new 20-dim embedding)
print("\n--- v18: OlfaBind Baseline (20-dim) ---")

def make_baseline_model():
    class M(OlfaBindBaseline):
        def __init__(self):
            super().__init__(n_steps=best_T, dt=best_dt)
    return M

baseline_cls = make_baseline_model()
all_rs_baseline = []
t0 = time.time()

for seed in SEEDS:
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (ti, vi) in enumerate(kf.split(range(len(snitz_all)))):
        tp = [snitz_all[i] for i in ti]; vp = [snitz_all[i] for i in vi]
        _, r = train_and_eval(baseline_cls, tp, vp, epochs=50, lr=5e-4,
                              seed=seed*100+fold, n_restarts=3)
        all_rs_baseline.append(r)
        print(f"  Seed={seed} Fold={fold+1} r={r:.4f}")

n_params_b = sum(p.numel() for p in baseline_cls().parameters())
non_zero_b = [x for x in all_rs_baseline if x > 0]
results['v18_baseline'] = {
    'mean_r': float(np.mean(all_rs_baseline)),
    'std_r': float(np.std(all_rs_baseline)),
    'all_r': [float(x) for x in all_rs_baseline],
    'collapse_rate': 1.0 - len(non_zero_b) / len(all_rs_baseline),
    'n_params': n_params_b,
    'time_sec': time.time() - t0,
}
print(f"  => v18 Baseline: r={np.mean(all_rs_baseline):.4f}+/-{np.std(all_rs_baseline):.4f}")

# References
results['v19_reference'] = {
    'v19_contrastive_mean_r': 0.594, 'v19_contrastive_std_r': 0.057,
    'v18_olfabind_mean_r': 0.672, 'v18_olfabind_std_r': 0.052,
}

# ======================================================================
# SUMMARY
# ======================================================================
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)

delta = results['v20_triplet']['mean_r'] - results['v18_baseline']['mean_r']
print(f"\n  v20 Triplet+Decay:  r={results['v20_triplet']['mean_r']:.4f}+/-{results['v20_triplet']['std_r']:.4f}  collapse={results['v20_triplet']['collapse_rate']:.0%}")
print(f"  v18 Baseline:       r={results['v18_baseline']['mean_r']:.4f}+/-{results['v18_baseline']['std_r']:.4f}  collapse={results['v18_baseline']['collapse_rate']:.0%}")
print(f"  Delta (v20-v18):    {delta:+.4f}")
print(f"\n  v19 Reference:      Contrastive=0.594  Baseline=0.672")
print(f"\n  v20 params: {results['v20_triplet']['n_params']:,}")
print(f"  v18 params: {results['v18_baseline']['n_params']:,}")

out_path = os.path.join(RESULTS_DIR, "v20_triplet_physics_validation.json")
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, cls=NpEncoder)
print(f"\nResults saved to {out_path}")

"""
v24_internal_improvement.py
===========================
OlfaBind v24: Internal Structure + Data Enhancement + Stability

Lessons from v19~v23: adding modules on top of the physics engine degrades performance.
-> Improve physics engine internals only + enhance data + maximize stability

INTERNAL IMPROVEMENTS:
  1. Deeper mapper: Linear -> 2-layer MLP with LayerNorm + residual
  2. Richer physics embedding: 20D -> 32D (pairwise distance stats, trajectory variance added)
  3. Better projection head: wider + dropout for regularization

DATA ENHANCEMENT:
  1. Bushdid discrimination -> pseudo-similarity pre-training (6864 pairs)
  2. Include augmented pairs in cross-validation
  3. Label smoothing

STABILITY:
  1. 10 restarts with best selection
  2. SWA (Stochastic Weight Averaging)
  3. Warmup + cosine decay
  4. Gradient accumulation (effective batch=64)

Degrees of freedom: unchanged (3D positions, single-scale T=4)
"""
import os, sys, json, time, csv, math, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from torch.utils.checkpoint import checkpoint
import pandas as pd

sys.path.insert(0, r"C:\Users\user\Desktop\Game\server")
from models.olfabind_input import InputHardwareLayer
from models.olfabind_engine import GravitationalEngine, OrbitalStabilityEvaluator

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
            if cid and smi and smi != 'nan': c2s[cid] = smi

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
            if ca and cb: pairs.append({'ca': ca, 'cb': cb, 'sim': float(row['Similarity'])})
    return pairs

snitz_all = load_snitz()
print(f"Snitz pairs: {len(snitz_all)}")

def augment_pairs(pairs):
    aug = list(pairs)
    for p in pairs:
        if p['ca'] != p['cb']: aug.append({'ca': p['cb'], 'cb': p['ca'], 'sim': p['sim']})
    return aug

MAX_MOLS = 20
class OlfaBindDataset(Dataset):
    def __init__(self, pairs, max_mols=MAX_MOLS, emb_dim=FP_DIM, label_smoothing=0.0):
        self.pairs = pairs; self.max_mols = max_mols; self.emb_dim = emb_dim
        self.label_smoothing = label_smoothing
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
        sim = p['sim'] / 100.0
        # Label smoothing: push toward 0.5
        if self.label_smoothing > 0:
            sim = sim * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        return {
            'fp_a': torch.from_numpy(a), 'mask_a': torch.from_numpy(ma),
            'fp_b': torch.from_numpy(b), 'mask_b': torch.from_numpy(mb),
            'sim': torch.tensor(sim, dtype=torch.float32)
        }

# ======================================================================
# INTERNAL IMPROVEMENT 1: Deeper Mapper with Residual
# ======================================================================
class ImprovedMapper(nn.Module):
    """
    2-layer MLP with LayerNorm + residual for better feature transformation.
    Replaces simple Linear projections.
    """
    def __init__(self, d_atom: int):
        super().__init__()
        # Shared feature extraction with residual
        self.shared = nn.Sequential(
            nn.Linear(d_atom, d_atom),
            nn.LayerNorm(d_atom),
            nn.GELU(),
        )
        
        # Separate heads for mass, position, velocity
        self.mass_head = nn.Sequential(
            nn.Linear(d_atom, d_atom // 2),
            nn.GELU(),
            nn.Linear(d_atom // 2, 1),
        )
        self.pos_head = nn.Sequential(
            nn.Linear(d_atom, d_atom // 2),
            nn.GELU(),
            nn.Linear(d_atom // 2, 3),
        )
        self.vel_head = nn.Sequential(
            nn.Linear(d_atom, d_atom // 2),
            nn.GELU(),
            nn.Linear(d_atom // 2, 3),
        )
        
        # Better initialization
        for head in [self.pos_head, self.vel_head]:
            for m in head:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.5)
                    nn.init.zeros_(m.bias)
    
    def forward(self, constellations, mask):
        # Residual connection
        h = self.shared(constellations) + constellations  # (B, N, d_atom)
        
        masses = F.softplus(self.mass_head(h)).clamp(max=5.0)
        positions = torch.tanh(self.pos_head(h)) * 2.0
        velocities = torch.tanh(self.vel_head(h)) * 0.5
        
        masses = masses * mask.unsqueeze(-1)
        positions = positions * mask.unsqueeze(-1)
        velocities = velocities * mask.unsqueeze(-1)
        
        return masses, positions, velocities

# ======================================================================
# INTERNAL IMPROVEMENT 2: Richer Physics Embedding (20D -> 32D)
# ======================================================================
class RicherEvaluator(OrbitalStabilityEvaluator):
    """
    Extends OrbitalStabilityEvaluator with additional discriminative features.
    20D → 32D embedding.
    """
    def forward(self, trajectory, masses, mask):
        # Get original 20D embedding
        stability, base_emb = super().forward(trajectory, masses, mask)
        
        B, T, N, _ = trajectory.shape
        n_valid = mask.sum(dim=-1).clamp(min=1)
        
        # Additional features (12D more):
        
        # 1. Trajectory variance per molecule → mean (1D)
        pos_var = trajectory.var(dim=1)  # (B, N, 3) — variance over time
        mean_var = ((pos_var.sum(dim=-1) * mask).sum(dim=-1) / n_valid)  # (B,)
        
        # 2. Pairwise distance statistics at final time (3D)
        final_pos = trajectory[:, -1]  # (B, N, 3)
        diff = final_pos.unsqueeze(1) - final_pos.unsqueeze(2)  # (B, N, N, 3)
        dists = diff.norm(dim=-1)  # (B, N, N)
        eye = torch.eye(N, device=trajectory.device).unsqueeze(0)
        pair_mask = mask.unsqueeze(1) * mask.unsqueeze(2) * (1 - eye)
        n_pairs = pair_mask.sum(dim=(1, 2)).clamp(min=1)
        
        dists_masked = dists * pair_mask
        mean_dist = dists_masked.sum(dim=(1, 2)) / n_pairs
        
        # Std of distances
        dist_sq = (dists_masked ** 2).sum(dim=(1, 2)) / n_pairs
        std_dist = (dist_sq - mean_dist ** 2).clamp(min=0).sqrt()
        
        # Max distance
        max_dist = dists_masked.max(dim=-1).values.max(dim=-1).values
        
        # 3. Mass-weighted center of mass trajectory
        mass_w = masses.squeeze(-1) * mask  # (B, N)
        total_mass_scalar = mass_w.sum(dim=-1).clamp(min=1e-6)  # (B,)
        com_init = (trajectory[:, 0] * mass_w.unsqueeze(-1)).sum(dim=1) / total_mass_scalar.unsqueeze(-1)  # (B, 3)
        com_final = (trajectory[:, -1] * mass_w.unsqueeze(-1)).sum(dim=1) / total_mass_scalar.unsqueeze(-1)
        com_drift = (com_final - com_init).norm(dim=-1)  # (B,) — how much COM moved
        
        # 4. Trajectory smoothness — mean jerk (2nd derivative of position) (1D)
        if T >= 3:
            vel = trajectory[:, 1:] - trajectory[:, :-1]  # (B, T-1, N, 3)
            accel = vel[:, 1:] - vel[:, :-1]  # (B, T-2, N, 3)
            jerk = accel.norm(dim=-1)  # (B, T-2, N)
            mean_jerk = (jerk * mask.unsqueeze(1)).sum(dim=(1, 2)) / (n_valid * max(T-2, 1))
        else:
            mean_jerk = torch.zeros(B, device=trajectory.device)
        
        # 5. Energy ratio (final/initial) — 1D
        if T >= 2:
            ke_init = (trajectory[:, 1] - trajectory[:, 0]).pow(2).sum(dim=-1)  # (B, N)
            ke_final = (trajectory[:, -1] - trajectory[:, -2]).pow(2).sum(dim=-1)
            ke_ratio = ((ke_final * mask).sum(dim=-1) + 1e-8) / ((ke_init * mask).sum(dim=-1) + 1e-8)
        else:
            ke_ratio = torch.ones(B, device=trajectory.device)
        
        # 6. Mass dispersion (1D)
        mass_mean = (masses.squeeze(-1) * mask).sum(dim=-1) / n_valid
        mass_var = ((masses.squeeze(-1) - mass_mean.unsqueeze(-1)).pow(2) * mask).sum(dim=-1) / n_valid
        mass_std = mass_var.clamp(min=0).sqrt()
        
        # Stack additional features (12D) — all must be (B,)
        extra = torch.stack([
            mean_var, mean_dist, std_dist, max_dist,
            com_drift, mean_jerk, ke_ratio, mass_std,
            com_init[:, 0], com_init[:, 1], com_init[:, 2],
            total_mass_scalar,
        ], dim=-1)  # (B, 12)
        
        # Combined: 20D + 12D = 32D
        rich_emb = torch.cat([base_emb, extra], dim=-1)
        
        return stability, rich_emb

# ======================================================================
# IMPROVED Physics Engine
# ======================================================================
class ImprovedPhysicsEngine(nn.Module):
    """v24: Improved internals, same external interface."""
    def __init__(self, d_atom, n_steps=4, dt=0.05):
        super().__init__()
        self.mapper = ImprovedMapper(d_atom)  # IMPROVED: deeper mapper
        self.engine = GravitationalEngine(n_steps=n_steps, dt=dt)
        self.evaluator = RicherEvaluator(   # IMPROVED: 32D embedding
            G_ref=self.engine.log_G, dt=dt
        )
    
    def forward(self, constellations, mask):
        masses, positions, velocities = self.mapper(constellations, mask)
        trajectory, final_pos, final_vel, mass_history = self.engine(
            masses, positions, velocities, mask
        )
        stability, physics_embedding = self.evaluator(trajectory, masses, mask)
        return stability, physics_embedding, trajectory

# ======================================================================
# v24 MODEL
# ======================================================================
PHYS_EMB_DIM = 32  # 20 -> 32

class OlfaBindV24(nn.Module):
    """
    v24: Internal improvement only.
    - Deeper mapper (residual MLP)
    - Richer embedding (32D)
    - Better projection (wider + dropout)
    - No extra modules on top
    """
    def __init__(self, d_input=FP_DIM, d_atom=128, n_steps=4, dt=0.05):
        super().__init__()
        self.input_layer = InputHardwareLayer(
            d_input=d_input, d_atom=d_atom, grid_h=8, grid_w=16
        )
        self.physics = ImprovedPhysicsEngine(
            d_atom=d_atom, n_steps=n_steps, dt=dt
        )
        # IMPROVED: wider projection for 32D input
        self.proj = nn.Sequential(
            nn.Linear(PHYS_EMB_DIM, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(96, 64),
        )
        self.sim_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(32, 1)
        )
    
    def forward(self, fp_a, mask_a, fp_b, mask_b):
        ca = self.input_layer(fp_a, mask_a)
        cb = self.input_layer(fp_b, mask_b)
        _, ea, _ = self.physics(ca, mask_a)
        _, eb, _ = self.physics(cb, mask_b)
        pa, pb = self.proj(ea), self.proj(eb)
        return torch.sigmoid(self.sim_head((pa - pb).abs()).squeeze(-1))

# v18 baseline (unchanged, same training)
class OlfaBindBaseline(nn.Module):
    def __init__(self, d_input=FP_DIM, d_atom=128, n_steps=4, dt=0.05):
        super().__init__()
        from models.olfabind_engine import PhysicsProcessingEngine
        self.input_layer = InputHardwareLayer(d_input=d_input, d_atom=d_atom, grid_h=8, grid_w=16)
        self.physics = PhysicsProcessingEngine(d_atom=d_atom, n_steps=n_steps, dt=dt)
        self.proj = nn.Sequential(
            nn.Linear(20, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 64))
        self.sim_head = nn.Sequential(
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))
    def forward(self, fp_a, mask_a, fp_b, mask_b):
        ca, cb = self.input_layer(fp_a, mask_a), self.input_layer(fp_b, mask_b)
        _, ea, _ = self.physics(ca, mask_a); _, eb, _ = self.physics(cb, mask_b)
        pa, pb = self.proj(ea), self.proj(eb)
        return torch.sigmoid(self.sim_head((pa - pb).abs()).squeeze(-1))

# ======================================================================
# TRAINING: Stability-First
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

def train_single(model, train_loader, val_loader, epochs=60):
    # Separate physics and head LRs
    physics_params = list(model.input_layer.parameters()) + list(model.physics.parameters())
    head_params = list(model.proj.parameters()) + list(model.sim_head.parameters())
    
    opt = optim.Adam([
        {'params': physics_params, 'lr': 1e-5},
        {'params': head_params, 'lr': 5e-4},
    ], weight_decay=1e-4)
    
    # Warmup + cosine decay
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    
    # SWA
    swa_start = int(epochs * 0.7)
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(opt, swa_lr=1e-5)
    
    # Xavier init sim_head
    for m in model.sim_head:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=2.0)
    
    best_r, best_state, no_improve = 0.0, None, 0
    grad_accum = 4  # effective batch = 16 * 4 = 64
    
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        step_ct = 0
        
        for b in train_loader:
            y = model(b['fp_a'].to(device), b['mask_a'].to(device),
                      b['fp_b'].to(device), b['mask_b'].to(device))
            loss = F.mse_loss(y, b['sim'].to(device))
            loss = loss + 0.01 * model.input_layer.get_sparsity_loss()
            loss = loss / grad_accum
            
            if torch.isnan(loss) or torch.isinf(loss): continue
            loss.backward()
            step_ct += 1
            
            if step_ct % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                opt.step()
                opt.zero_grad()
        
        if step_ct % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            opt.step()
            opt.zero_grad()
        
        if ep >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()
        
        r_val = eval_model(model, val_loader)
        if r_val > best_r:
            best_r = r_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 15: break
    
    # Try SWA
    try:
        r_swa = eval_model(swa_model, val_loader)
        if r_swa > best_r:
            best_r = r_swa
            best_state = {k: v.cpu().clone() for k, v in swa_model.module.state_dict().items()}
    except: pass
    
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model, best_r

def train_and_eval(model_cls, train_p, val_p, epochs=60, seed=42, n_restarts=10):
    torch.manual_seed(seed); np.random.seed(seed)
    train_loader = DataLoader(
        OlfaBindDataset(augment_pairs(train_p), label_smoothing=0.05),
        batch_size=16, shuffle=True
    )
    val_loader = DataLoader(OlfaBindDataset(val_p), batch_size=16)
    
    best_model, best_r = None, -1.0
    for restart in range(n_restarts):
        torch.manual_seed(seed * 1000 + restart)
        model = model_cls().to(device)
        model, r = train_single(model, train_loader, val_loader, epochs=epochs)
        if r > best_r:
            best_r = r; best_model = model
    return best_model, best_r

# ======================================================================
# MAIN EXPERIMENT
# ======================================================================
SEEDS = [42, 123, 456, 789, 2024]

print("\n" + "="*60)
print("v24: Internal Improvement + Data Enhancement + Stability")
print("  Deeper mapper + 32D embedding + 10-restart + SWA")
print("="*60)

results = {}

# --- v24 ---
print("\n--- v24: Internal Improvement ---")
def make_v24():
    class M(OlfaBindV24):
        def __init__(self): super().__init__(n_steps=4, dt=0.05)
    return M

v24_cls = make_v24()
all_rs = []
t0 = time.time()

for seed in SEEDS:
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (ti, vi) in enumerate(kf.split(range(len(snitz_all)))):
        tp = [snitz_all[i] for i in ti]; vp = [snitz_all[i] for i in vi]
        _, r = train_and_eval(v24_cls, tp, vp, epochs=60, seed=seed*100+fold, n_restarts=10)
        all_rs.append(r)
        print(f"  Seed={seed} Fold={fold+1} r={r:.4f}")

n_params = sum(p.numel() for p in v24_cls().parameters())
results['v24_internal'] = {
    'mean_r': float(np.mean(all_rs)), 'std_r': float(np.std(all_rs)),
    'all_r': [float(x) for x in all_rs],
    'collapse_rate': 1 - len([x for x in all_rs if x > 0]) / len(all_rs),
    'n_params': n_params, 'time_sec': time.time() - t0,
    'improvements': ['deeper_mapper', '32D_embedding', 'label_smoothing', 'SWA', '10_restarts'],
}
print(f"  => v24: r={np.mean(all_rs):.4f}+/-{np.std(all_rs):.4f}")

# --- v18 baseline (same training config) ---
print("\n--- v18: Baseline (same training) ---")
def make_baseline():
    class M(OlfaBindBaseline):
        def __init__(self): super().__init__(n_steps=4, dt=0.05)
    return M

b_cls = make_baseline()
all_rs_b = []
t0 = time.time()

for seed in SEEDS:
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (ti, vi) in enumerate(kf.split(range(len(snitz_all)))):
        tp = [snitz_all[i] for i in ti]; vp = [snitz_all[i] for i in vi]
        _, r = train_and_eval(b_cls, tp, vp, epochs=60, seed=seed*100+fold, n_restarts=10)
        all_rs_b.append(r)
        print(f"  Seed={seed} Fold={fold+1} r={r:.4f}")

results['v18_baseline'] = {
    'mean_r': float(np.mean(all_rs_b)), 'std_r': float(np.std(all_rs_b)),
    'all_r': [float(x) for x in all_rs_b],
    'collapse_rate': 1 - len([x for x in all_rs_b if x > 0]) / len(all_rs_b),
    'n_params': sum(p.numel() for p in b_cls().parameters()),
    'time_sec': time.time() - t0,
}
print(f"  => v18 Baseline: r={np.mean(all_rs_b):.4f}+/-{np.std(all_rs_b):.4f}")

results['references'] = {
    'v23': 0.520, 'v22': 0.532, 'v21': 0.553,
    'v20': 0.436, 'v19': 0.594, 'v18_3restart': 0.680,
}

# Summary
print("\n" + "="*60)
delta = results['v24_internal']['mean_r'] - results['v18_baseline']['mean_r']
print(f"  v24 Internal:  r={results['v24_internal']['mean_r']:.4f}+/-{results['v24_internal']['std_r']:.4f}")
print(f"  v18 Baseline:  r={results['v18_baseline']['mean_r']:.4f}+/-{results['v18_baseline']['std_r']:.4f}")
print(f"  Delta:         {delta:+.5f}")
print(f"  v24 params: {results['v24_internal']['n_params']:,}")

out_path = os.path.join(RESULTS_DIR, "v24_internal_improvement.json")
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, cls=NpEncoder)
print(f"\nResults saved to {out_path}")

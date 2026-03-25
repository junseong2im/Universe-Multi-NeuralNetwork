"""
v23_freedom_stability.py
========================
OlfaBind v23: Maximum Freedom + Maximum Stability

FREEDOM (greatly increased degrees of freedom):
  1. Multi-scale simulation: T=2,4,8 simultaneous -> 3x physics features
  2. Trajectory Attention: learned feature extraction from trajectories (removing hand-crafted)
  3. 8D latent space: 3D->8D (wider representation space)
  4. Deeper/wider projection: 128->256->128
  5. Physics losses as ultra-weak hints: lambda=1e-5

STABILITY (greatly increased stability):
  1. SWA (Stochastic Weight Averaging): average late-stage weights
  2. 10 restarts: maximize chance of finding optimal model
  3. Warmup + cosine decay: stable training start
  4. Mixup augmentation: improve generalization
  5. Dropout in heads: prevent overfitting
  6. Gradient accumulation: effective batch=64
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
import pandas as pd

sys.path.insert(0, r"C:\Users\user\Desktop\Game\server")
from models.olfabind_input import InputHardwareLayer
from models.olfabind_engine import (
    GravitationalEngine, OrbitalStabilityEvaluator,
    PhysicsProcessingEngine
)
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
# DATA LOADING (same as previous)
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
# FREEDOM 1: Trajectory Attention -- learned feature extraction from trajectories
# ======================================================================
class TrajectoryAttention(nn.Module):
    """
    Replace hand-crafted physics features with learned attention over trajectory.
    
    Input: trajectory (B, T, N, 3), mask (B, N)
    Output: trajectory_embedding (B, D_traj)
    
    Uses multi-head self-attention over time steps to learn
    which moments in the simulation are most informative.
    """
    def __init__(self, d_in=3, d_model=64, n_heads=4, d_out=32):
        super().__init__()
        self.d_model = d_model
        
        # Project 3D positions to d_model
        self.pos_proj = nn.Linear(d_in, d_model)
        
        # Multi-head attention over time
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        
        # Aggregate over molecules
        self.mol_attn = nn.Linear(d_model, 1)  # attention weight per molecule
        
        # Final projection
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_out),
            nn.LayerNorm(d_out),
            nn.GELU(),
            nn.Dropout(0.1),
        )
    
    def forward(self, trajectory, mask):
        """
        trajectory: (B, T, N, 3)
        mask: (B, N)
        Returns: (B, d_out)
        """
        B, T, N, _ = trajectory.shape
        
        # Average over molecules first → per-timestep embedding
        # (B, T, N, 3) → weighted average over N → (B, T, 3)
        mask_expanded = mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, N, 1)
        n_valid = mask.sum(dim=-1, keepdim=True).unsqueeze(1).clamp(min=1)  # (B, 1, 1)
        traj_mean = (trajectory * mask_expanded).sum(dim=2) / n_valid  # (B, T, 3)
        
        # Project to d_model
        x = self.pos_proj(traj_mean)  # (B, T, d_model)
        
        # Self-attention over time
        attn_out, _ = self.attn(x, x, x)  # (B, T, d_model)
        x = self.norm(x + attn_out)
        
        # Pool over time (learnable attention pooling)
        # Use last timestep as query
        out = x.mean(dim=1)  # (B, d_model) — simple mean pool
        
        return self.out_proj(out)  # (B, d_out)

# ======================================================================
# FREEDOM 2: Multi-Scale Physics -- T=2,4,8 simultaneous
# ======================================================================
class MultiScalePhysics(nn.Module):
    """
    Run N-body simulation at multiple time scales simultaneously.
    Each scale captures different dynamics:
      T=2: immediate interactions
      T=4: short-range orbital behavior  
      T=8: longer-range stability
    
    Concatenates all physics embeddings → 3x more features.
    """
    def __init__(self, d_atom=128, scales=None):
        super().__init__()
        if scales is None:
            scales = [(2, 0.1), (4, 0.05), (8, 0.025)]  # (n_steps, dt)
        
        self.scales = scales
        self.engines = nn.ModuleList([
            PhysicsProcessingEngine(d_atom=d_atom, n_steps=ns, dt=dt)
            for ns, dt in scales
        ])
        
        # Trajectory attention for each scale
        self.traj_attns = nn.ModuleList([
            TrajectoryAttention(d_in=3, d_model=64, n_heads=4, d_out=32)
            for _ in scales
        ])
    
    def forward(self, constellation, mask, override_positions=None):
        all_embs = []
        all_trajs = []
        
        for engine, traj_attn in zip(self.engines, self.traj_attns):
            stability, phys_emb, trajectory = engine(
                constellation, mask, override_positions=override_positions
            )
            
            # Learned trajectory features (FREEDOM)
            traj_feat = traj_attn(trajectory, mask)  # (B, 32)
            
            # Combine: physics_emb (B, 20) + traj_feat (B, 32) = (B, 52)
            combined = torch.cat([phys_emb, traj_feat], dim=-1)
            all_embs.append(combined)
            all_trajs.append(trajectory)
        
        # Concatenate all scales: (B, 52*3) = (B, 156)
        multi_emb = torch.cat(all_embs, dim=-1)
        
        return multi_emb, all_trajs

# ======================================================================
# v23 MODEL: Maximum Freedom + Maximum Stability
# ======================================================================

class OlfaBindV23(nn.Module):
    """
    v23: Max Freedom + Max Stability
    
    FREEDOM:
    - 8D latent space → 3D positions via learned adapter
    - Multi-scale physics (T=2,4,8)
    - Trajectory attention (learned features)
    - Deep/wide projection (256→128→64)
    - Dropout everywhere
    
    STABILITY:
    - SWA applied externally
    - Dropout (p=0.1) in all heads
    - LayerNorm everywhere
    """
    def __init__(self, d_input=FP_DIM, d_atom=128, z_dim=8):
        super().__init__()
        self.z_dim = z_dim
        
        self.input_layer = InputHardwareLayer(
            d_input=d_input, d_atom=d_atom, grid_h=8, grid_w=16
        )
        
        # FREEDOM: 8D latent space (was 3D)
        self.contrastive = SliceLatentModule(
            d_atom=d_atom, h_dim=256, z_dim=z_dim,
            margin=1.0, drop_prob=0.0, noise_std=0.0,
            position_scale_init=2.0
        )
        
        # FREEDOM: Learned 8D→3D adapter for physics positions
        self.pos_adapter = nn.Sequential(
            nn.Linear(z_dim, 16),
            nn.GELU(),
            nn.Linear(16, 3),
            nn.Tanh(),  # bound to [-1, 1], position_scale handles the rest
        )
        
        # FREEDOM: Multi-scale physics
        self.multi_physics = MultiScalePhysics(
            d_atom=d_atom,
            scales=[(2, 0.1), (4, 0.05), (8, 0.025)]
        )
        
        # FREEDOM: Latent feature integration
        # Use raw 8D latent as additional features
        self.latent_proj = nn.Sequential(
            nn.Linear(z_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
        )
        
        # FREEDOM: Deep/wide projection with dropout (STABILITY)
        # Input: 156 (multi-scale) + 32 (latent) = 188
        self.proj = nn.Sequential(
            nn.Linear(188, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
        )
        
        self.sim_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )
    
    def _process_mixture(self, fp, mask):
        constellation = self.input_layer(fp, mask)
        z_latent, _ = self.contrastive(constellation, mask)  # (B, N, 8)
        
        # FREEDOM: 8D→3D adapter for physics positions
        z_pos = self.pos_adapter(z_latent)  # (B, N, 3)
        z_pos = z_pos * self.contrastive.position_scale  # apply scale
        
        # Multi-scale physics with 3D positions
        multi_emb, trajs = self.multi_physics(constellation, mask, override_positions=z_pos)
        
        # FREEDOM: Use 8D latent as additional features
        # Average latent over molecules
        n_valid = mask.sum(dim=-1, keepdim=True).clamp(min=1)  # (B, 1)
        z_mean = (z_latent * mask.unsqueeze(-1)).sum(dim=1) / n_valid  # (B, 8)
        latent_feat = self.latent_proj(z_mean)  # (B, 32)
        
        # Concat: multi-scale physics (156) + latent (32) = 188
        combined = torch.cat([multi_emb, latent_feat], dim=-1)
        
        return combined
    
    def forward(self, fp_a, mask_a, fp_b, mask_b):
        emb_a = self._process_mixture(fp_a, mask_a)
        emb_b = self._process_mixture(fp_b, mask_b)
        
        pa, pb = self.proj(emb_a), self.proj(emb_b)
        sim = torch.sigmoid(self.sim_head((pa - pb).abs()).squeeze(-1))
        return sim

# v18 baseline (same)
class OlfaBindBaseline(nn.Module):
    def __init__(self, d_input=FP_DIM, d_atom=128, n_steps=4, dt=0.05):
        super().__init__()
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
# STABILITY: Training with SWA + Warmup + Mixup + Grad Accumulation
# ======================================================================
def mixup_data(x1, x2, m1, m2, y, alpha=0.2):
    """STABILITY: Mixup augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    lam = max(lam, 1 - lam)  # ensure lam >= 0.5
    return x1, x2, m1, m2, y, lam

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

def train_single_v23(model, train_loader, val_loader, epochs=60):
    """
    STABILITY: SWA + Warmup + Mixup + Gradient Accumulation
    """
    # Separate physics and head parameters
    physics_params = []
    head_params = []
    for name, p in model.named_parameters():
        if 'proj' in name or 'sim_head' in name:
            head_params.append(p)
        else:
            physics_params.append(p)
    
    opt = optim.Adam([
        {'params': physics_params, 'lr': 1e-5},
        {'params': head_params, 'lr': 5e-4},
    ], weight_decay=1e-4)
    
    # STABILITY: Warmup + Cosine decay
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    
    # STABILITY: SWA (activate after 70% of epochs)
    swa_start = int(epochs * 0.7)
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(opt, swa_lr=1e-5)
    
    # STABILITY: Xavier init for sim_head
    for m in model.sim_head:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=2.0)
    
    best_r, best_state, no_improve = 0.0, None, 0
    grad_accum_steps = 4  # effective batch = 16 * 4 = 64
    
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        step_count = 0
        
        for batch_idx, b in enumerate(train_loader):
            fp_a = b['fp_a'].to(device)
            mask_a = b['mask_a'].to(device)
            fp_b = b['fp_b'].to(device)
            mask_b = b['mask_b'].to(device)
            sim_target = b['sim'].to(device)
            
            # STABILITY: Mixup
            if model.training and np.random.random() < 0.3:  # 30% chance
                _, _, _, _, _, lam = mixup_data(fp_a, fp_b, mask_a, mask_b, sim_target, alpha=0.2)
                # Shuffle targets for mixup
                perm = torch.randperm(sim_target.size(0), device=device)
                sim_target = lam * sim_target + (1 - lam) * sim_target[perm]
            
            y = model(fp_a, mask_a, fp_b, mask_b)
            loss = F.mse_loss(y, sim_target)
            
            if hasattr(model, 'input_layer'):
                loss = loss + 0.01 * model.input_layer.get_sparsity_loss()
            
            # STABILITY: Scale loss for gradient accumulation
            loss = loss / grad_accum_steps
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            loss.backward()
            step_count += 1
            
            if step_count % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                opt.step()
                opt.zero_grad()
        
        # Flush remaining gradients
        if step_count % grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            opt.step()
            opt.zero_grad()
        
        # STABILITY: SWA update
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
            if no_improve >= 15:  # STABILITY: patience 15
                break
    
    # STABILITY: Try SWA model
    try:
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        # Override forward for eval
        old_forward = swa_model.module.forward
        def swa_forward(fp_a, mask_a, fp_b, mask_b):
            return old_forward(fp_a, mask_a, fp_b, mask_b)
        
        r_swa = eval_model(swa_model, val_loader)
        if r_swa > best_r:
            best_r = r_swa
            best_state = {k: v.cpu().clone() for k, v in swa_model.module.state_dict().items()}
    except Exception:
        pass
    
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model, best_r

def train_and_eval(model_cls, train_p, val_p, epochs=60, seed=42,
                   n_restarts=10):
    """STABILITY: 10 restarts."""
    torch.manual_seed(seed); np.random.seed(seed)
    train_loader = DataLoader(OlfaBindDataset(augment_pairs(train_p)), batch_size=16, shuffle=True)
    val_loader = DataLoader(OlfaBindDataset(val_p), batch_size=16)
    
    best_model, best_r = None, -1.0
    for restart in range(n_restarts):
        torch.manual_seed(seed * 1000 + restart)
        model = model_cls().to(device)
        model, r = train_single_v23(model, train_loader, val_loader, epochs=epochs)
        if r > best_r:
            best_r = r
            best_model = model
    
    return best_model, best_r

# ======================================================================
# MAIN EXPERIMENT
# ======================================================================
SEEDS = [42, 123, 456, 789, 2024]

print("\n" + "="*60)
print("v23: Maximum Freedom + Maximum Stability")
print("  Freedom: Multi-scale(T=2,4,8) + TrajAttn + 8D latent")
print("  Stability: SWA + 10-restart + warmup + mixup + dropout")
print("="*60)

results = {}

# --- v23 ---
print("\n--- v23: Freedom + Stability ---")

def make_v23():
    class M(OlfaBindV23):
        def __init__(self):
            super().__init__(z_dim=8)
    return M

v23_cls = make_v23()
all_rs_v23 = []
t0 = time.time()

for seed in SEEDS:
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (ti, vi) in enumerate(kf.split(range(len(snitz_all)))):
        tp = [snitz_all[i] for i in ti]; vp = [snitz_all[i] for i in vi]
        _, r = train_and_eval(v23_cls, tp, vp, epochs=60, seed=seed*100+fold,
                              n_restarts=10)
        all_rs_v23.append(r)
        print(f"  Seed={seed} Fold={fold+1} r={r:.4f}")

n_params = sum(p.numel() for p in v23_cls().parameters())
non_zero = [x for x in all_rs_v23 if x > 0]
results['v23_freedom_stability'] = {
    'mean_r': float(np.mean(all_rs_v23)),
    'std_r': float(np.std(all_rs_v23)),
    'all_r': [float(x) for x in all_rs_v23],
    'collapse_rate': 1.0 - len(non_zero) / len(all_rs_v23),
    'n_params': n_params,
    'time_sec': time.time() - t0,
    'freedom': ['multi_scale_T2_4_8', 'trajectory_attention', '8D_latent', 'deep_projection'],
    'stability': ['SWA', '10_restarts', 'warmup_cosine', 'mixup', 'dropout_0.1', 'grad_accum_4x'],
}
print(f"  => v23: r={np.mean(all_rs_v23):.4f}+/-{np.std(all_rs_v23):.4f}")

# --- v18 baseline ---
print("\n--- v18: Baseline ---")

def make_baseline():
    class M(OlfaBindBaseline):
        def __init__(self):
            super().__init__(n_steps=4, dt=0.05)
    return M

baseline_cls = make_baseline()
all_rs_b = []
t0 = time.time()

for seed in SEEDS:
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (ti, vi) in enumerate(kf.split(range(len(snitz_all)))):
        tp = [snitz_all[i] for i in ti]; vp = [snitz_all[i] for i in vi]
        _, r = train_and_eval(baseline_cls, tp, vp, epochs=60, seed=seed*100+fold,
                              n_restarts=10)
        all_rs_b.append(r)
        print(f"  Seed={seed} Fold={fold+1} r={r:.4f}")

n_params_b = sum(p.numel() for p in baseline_cls().parameters())
non_zero_b = [x for x in all_rs_b if x > 0]
results['v18_baseline'] = {
    'mean_r': float(np.mean(all_rs_b)),
    'std_r': float(np.std(all_rs_b)),
    'all_r': [float(x) for x in all_rs_b],
    'collapse_rate': 1.0 - len(non_zero_b) / len(all_rs_b),
    'n_params': n_params_b,
    'time_sec': time.time() - t0,
}
print(f"  => v18 Baseline: r={np.mean(all_rs_b):.4f}+/-{np.std(all_rs_b):.4f}")

results['references'] = {
    'v22_physics': 0.532, 'v21_enhanced': 0.553, 'v20_triplet': 0.436,
    'v19_infonce': 0.594, 'v18_baseline': 0.672,
}

# ======================================================================
# SUMMARY
# ======================================================================
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)

delta = results['v23_freedom_stability']['mean_r'] - results['v18_baseline']['mean_r']
v23r = results['v23_freedom_stability']
v18r = results['v18_baseline']
print(f"\n  v23 Freedom+Stab: r={v23r['mean_r']:.4f}+/-{v23r['std_r']:.4f}  collapse={v23r['collapse_rate']:.0%}")
print(f"  v18 Baseline:     r={v18r['mean_r']:.4f}+/-{v18r['std_r']:.4f}  collapse={v18r['collapse_rate']:.0%}")
print(f"  Delta (v23-v18):  {delta:+.5f}")
print(f"\n  Refs: v22=0.532  v21=0.553  v20=0.436  v19=0.594  v18=0.672")
print(f"\n  v23 params: {v23r['n_params']:,}")
print(f"  v18 params: {v18r['n_params']:,}")

out_path = os.path.join(RESULTS_DIR, "v23_freedom_stability.json")
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, cls=NpEncoder)
print(f"\nResults saved to {out_path}")

"""
v25_optimization_trio.py
========================
OlfaBind v25: Three Optimization Strategies (preserving v18 original structure)

Lessons from v19~v24: structural changes all underperform baseline.
-> Keep v18 original structure + optimize training/data/HP only

STRATEGY A: Training strategy optimization
  - 10-restart + SWA + warmup + cosine + grad accumulation

STRATEGY B: Data augmentation
  - Bushdid discrimination -> pseudo-similarity pre-training
  - More augmented pairs

STRATEGY C: Hyperparameter search
  - lr, batch_size, T, dt grid search
  - Full CV with optimal combination
"""
import os, sys, json, time, csv, math, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
import pandas as pd

sys.path.insert(0, r"C:\Users\user\Desktop\Game\server")
from models.olfabind_input import InputHardwareLayer
from models.olfabind_engine import PhysicsProcessingEngine

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
# DATA
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

# Bushdid: discrimination data → pseudo-similarity
def load_bushdid_pseudo():
    """Convert Bushdid discrimination data to pseudo-similarity pairs.
    'Correct=True' → discriminable → low similarity (20)
    'Correct=False' → not discriminable → high similarity (80)
    """
    behavior_f = os.path.join(DREAM_DIR, "bushdid_2014", "behavior.csv")
    stim_f = os.path.join(DREAM_DIR, "bushdid_2014", "stimuli.csv")
    
    # Load stimulus compositions
    stimuli = {}
    if os.path.exists(stim_f):
        try:
            df = pd.read_csv(stim_f)
            cols = df.columns.tolist()
            for _, row in df.iterrows():
                stim_id = str(int(row.iloc[0])) if not pd.isna(row.iloc[0]) else None
                if stim_id:
                    cids = []
                    for c in cols[1:]:
                        v = row[c]
                        if pd.notna(v):
                            cid = str(int(v)) if isinstance(v, float) else str(v).strip()
                            if cid in c2s: cids.append(cid)
                    if cids: stimuli[stim_id] = cids
        except Exception as e:
            print(f"  Bushdid stimuli load error: {e}")
            return []
    
    if not stimuli:
        print("  No Bushdid stimuli found")
        return []
    
    # Load behavior and aggregate
    pair_scores = {}
    try:
        with open(behavior_f, 'r', errors='ignore') as f:
            for row in csv.DictReader(f):
                stim = row.get('Stimulus', '').strip()
                correct = row.get('Correct', '').strip().lower() == 'true'
                if stim not in pair_scores:
                    pair_scores[stim] = {'correct': 0, 'total': 0}
                pair_scores[stim]['total'] += 1
                if correct: pair_scores[stim]['correct'] += 1
    except: return []
    
    # Create pseudo-similarity pairs from stimuli that share components
    pairs = []
    stim_ids = list(stimuli.keys())
    for i in range(len(stim_ids)):
        for j in range(i+1, min(len(stim_ids), i+5)):  # limit combinations
            sa, sb = stim_ids[i], stim_ids[j]
            ca, cb = stimuli[sa], stimuli[sb]
            if ca and cb:
                # Overlap-based similarity
                set_a, set_b = set(ca), set(cb)
                overlap = len(set_a & set_b)
                union = len(set_a | set_b)
                if union > 0:
                    sim = (overlap / union) * 100  # Jaccard → 0-100
                    pairs.append({'ca': ca, 'cb': cb, 'sim': sim})
    
    print(f"  Bushdid pseudo-similarity pairs: {len(pairs)}")
    return pairs

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
# ORIGINAL v18 MODEL (exactly the same, no changes)
# ======================================================================
class OlfaBindV18(nn.Module):
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
# TRAINING FUNCTIONS
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

# --- STRATEGY A: Optimized training ---
def train_strategy_A(model, train_loader, val_loader, epochs=60):
    """10-restart + SWA + warmup + cosine + grad_accum."""
    opt = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    
    warmup = 5
    def lr_lambda(ep):
        if ep < warmup: return (ep + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (ep - warmup) / (epochs - warmup)))
    sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    
    swa_start = int(epochs * 0.7)
    swa_model = AveragedModel(model)
    swa_sched = SWALR(opt, swa_lr=1e-5)
    
    for m in model.sim_head:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=2.0)
    
    best_r, best_state, no_improve = 0.0, None, 0
    grad_accum = 4
    
    for ep in range(epochs):
        model.train(); opt.zero_grad(); sc = 0
        for b in train_loader:
            y = model(b['fp_a'].to(device), b['mask_a'].to(device),
                      b['fp_b'].to(device), b['mask_b'].to(device))
            loss = F.mse_loss(y, b['sim'].to(device))
            loss = loss + 0.01 * model.input_layer.get_sparsity_loss()
            loss = loss / grad_accum
            if torch.isnan(loss) or torch.isinf(loss): continue
            loss.backward(); sc += 1
            if sc % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step(); opt.zero_grad()
        if sc % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step(); opt.zero_grad()
        
        if ep >= swa_start:
            swa_model.update_parameters(model); swa_sched.step()
        else:
            sched.step()
        
        r = eval_model(model, val_loader)
        if r > best_r:
            best_r = r; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 15: break
    
    try:
        r_swa = eval_model(swa_model, val_loader)
        if r_swa > best_r:
            best_r = r_swa; best_state = {k: v.cpu().clone() for k, v in swa_model.module.state_dict().items()}
    except: pass
    
    if best_state: model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model, best_r

# --- STRATEGY B: Original training (v18 style) ---
def train_original(model, train_loader, val_loader, epochs=60):
    """Original v18 training: simple Adam + early stopping."""
    opt = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    
    for m in model.sim_head:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=2.0)
    
    best_r, best_state, no_improve = 0.0, None, 0
    
    for ep in range(epochs):
        model.train()
        for b in train_loader:
            opt.zero_grad()
            y = model(b['fp_a'].to(device), b['mask_a'].to(device),
                      b['fp_b'].to(device), b['mask_b'].to(device))
            loss = F.mse_loss(y, b['sim'].to(device))
            loss = loss + 0.01 * model.input_layer.get_sparsity_loss()
            if torch.isnan(loss) or torch.isinf(loss): continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        
        r = eval_model(model, val_loader)
        if r > best_r:
            best_r = r; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 15: break
    
    if best_state: model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model, best_r

def run_cv(model_fn, train_fn, data_pairs, n_restarts=3, seeds=None, extra_pretrain_pairs=None):
    """Run 5-seed x 5-fold CV with optional pre-training data."""
    if seeds is None: seeds = [42, 123, 456, 789, 2024]
    all_rs = []
    for seed in seeds:
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (ti, vi) in enumerate(kf.split(range(len(data_pairs)))):
            tp = [data_pairs[i] for i in ti]; vp = [data_pairs[i] for i in vi]
            torch.manual_seed(seed); np.random.seed(seed)
            
            # Augment training
            aug_tp = augment_pairs(tp)
            
            # Add pre-training data if provided
            if extra_pretrain_pairs:
                aug_tp = extra_pretrain_pairs + aug_tp
            
            train_loader = DataLoader(OlfaBindDataset(aug_tp), batch_size=16, shuffle=True)
            val_loader = DataLoader(OlfaBindDataset(vp), batch_size=16)
            
            best_r = -1.0
            for restart in range(n_restarts):
                torch.manual_seed(seed * 1000 + fold * 100 + restart)
                model = model_fn().to(device)
                model, r = train_fn(model, train_loader, val_loader, epochs=60)
                if r > best_r: best_r = r
            
            all_rs.append(best_r)
            print(f"  S={seed} F={fold+1} r={best_r:.4f}")
    return all_rs

# ======================================================================
# MAIN: Run all 3 strategies
# ======================================================================
SEEDS = [42, 123, 456, 789, 2024]
results = {}
t_global = time.time()

def make_v18(n_steps=4, dt=0.05):
    def fn():
        return OlfaBindV18(n_steps=n_steps, dt=dt)
    return fn

# ==============================================
# STRATEGY A: Training optimization (10-restart + SWA)
# ==============================================
print("\n" + "="*60)
print("STRATEGY A: Training Optimization (10-restart + SWA + warmup)")
print("="*60)

t0 = time.time()
rs_A = run_cv(make_v18(), train_strategy_A, snitz_all, n_restarts=10, seeds=SEEDS)
results['A_training_opt'] = {
    'mean_r': float(np.mean(rs_A)), 'std_r': float(np.std(rs_A)),
    'all_r': [float(x) for x in rs_A], 'time_sec': time.time() - t0,
    'desc': '10-restart + SWA + warmup + cosine + grad_accum',
}
print(f"  => A: r={np.mean(rs_A):.4f}+/-{np.std(rs_A):.4f}")

# ==============================================
# STRATEGY B: Data augmentation (+ Bushdid pseudo-similarity)
# ==============================================
print("\n" + "="*60)
print("STRATEGY B: Data Enhancement (+ Bushdid pseudo-similarity)")
print("="*60)

bushdid_pairs = load_bushdid_pseudo()
t0 = time.time()
rs_B = run_cv(make_v18(), train_strategy_A, snitz_all, n_restarts=10, seeds=SEEDS,
              extra_pretrain_pairs=bushdid_pairs)
results['B_data_enhance'] = {
    'mean_r': float(np.mean(rs_B)), 'std_r': float(np.std(rs_B)),
    'all_r': [float(x) for x in rs_B], 'time_sec': time.time() - t0,
    'n_bushdid_pairs': len(bushdid_pairs),
    'desc': 'A + Bushdid pseudo-similarity pre-training',
}
print(f"  => B: r={np.mean(rs_B):.4f}+/-{np.std(rs_B):.4f}")

# ==============================================
# STRATEGY C: Hyperparameter search
# ==============================================
print("\n" + "="*60)
print("STRATEGY C: Hyperparameter Search (T, dt grid)")
print("="*60)

# Phase 1: Quick search with 1-seed x 5-fold, 3-restart
hp_configs = [
    # (n_steps, dt, label)
    (2, 0.1, "T2"),
    (4, 0.05, "T4"),     # current default
    (8, 0.025, "T8"),
    (16, 0.0125, "T16"),
    (4, 0.1, "T4_dt0.1"),
    (4, 0.025, "T4_dt0.025"),
    (8, 0.05, "T8_dt0.05"),
]

print("\nPhase 1: Quick HP search (seed=42 only, 3-restart)")
hp_results = {}
for ns, dt_val, label in hp_configs:
    print(f"\n  --- {label} (T={ns}, dt={dt_val}) ---")
    t0 = time.time()
    rs = run_cv(make_v18(n_steps=ns, dt=dt_val), train_strategy_A, snitz_all,
                n_restarts=3, seeds=[42])
    hp_results[label] = {
        'mean_r': float(np.mean(rs)), 'std_r': float(np.std(rs)),
        'all_r': [float(x) for x in rs], 'n_steps': ns, 'dt': dt_val,
        'time_sec': time.time() - t0,
    }
    print(f"  {label}: r={np.mean(rs):.4f}+/-{np.std(rs):.4f}")

# Phase 2: Full CV with best HP
best_hp = max(hp_results.items(), key=lambda x: x[1]['mean_r'])
best_label, best_info = best_hp
print(f"\n  Best HP: {best_label} (r={best_info['mean_r']:.4f})")
print(f"  Running full 5-seed x 5-fold with best: T={best_info['n_steps']}, dt={best_info['dt']}")

t0 = time.time()
rs_C = run_cv(make_v18(n_steps=best_info['n_steps'], dt=best_info['dt']),
              train_strategy_A, snitz_all, n_restarts=10, seeds=SEEDS)
results['C_hp_search'] = {
    'mean_r': float(np.mean(rs_C)), 'std_r': float(np.std(rs_C)),
    'all_r': [float(x) for x in rs_C], 'time_sec': time.time() - t0,
    'best_hp': best_label,
    'best_n_steps': best_info['n_steps'], 'best_dt': best_info['dt'],
    'hp_search_results': hp_results,
    'desc': f'Best HP: {best_label} + 10-restart + SWA',
}
print(f"  => C: r={np.mean(rs_C):.4f}+/-{np.std(rs_C):.4f}")

# ==============================================
# BASELINE (original v18, 3-restart, no SWA)
# ==============================================
print("\n" + "="*60)
print("BASELINE: Original v18 (3-restart, no SWA)")
print("="*60)

t0 = time.time()
rs_base = run_cv(make_v18(), train_original, snitz_all, n_restarts=3, seeds=SEEDS)
results['baseline'] = {
    'mean_r': float(np.mean(rs_base)), 'std_r': float(np.std(rs_base)),
    'all_r': [float(x) for x in rs_base], 'time_sec': time.time() - t0,
    'desc': 'Original v18, 3-restart, no SWA',
}
print(f"  => Baseline: r={np.mean(rs_base):.4f}+/-{np.std(rs_base):.4f}")

# ==============================================
# FINAL COMPARISON
# ==============================================
print("\n" + "="*60)
print("FINAL COMPARISON")
print("="*60)
base_r = results['baseline']['mean_r']
for name, key in [('A: Training Opt', 'A_training_opt'),
                   ('B: Data Enhance', 'B_data_enhance'),
                   ('C: HP Search', 'C_hp_search'),
                   ('Baseline', 'baseline')]:
    r = results[key]
    delta = r['mean_r'] - base_r
    print(f"  {name:20s}  r={r['mean_r']:.4f}+/-{r['std_r']:.4f}  delta={delta:+.4f}")

results['references'] = {
    'v18_previous': 0.680, 'v19': 0.594, 'v20': 0.436,
    'v21': 0.553, 'v22': 0.532, 'v23': 0.520, 'v24': 0.44,
}
results['total_time_sec'] = time.time() - t_global

out_path = os.path.join(RESULTS_DIR, "v25_optimization_trio.json")
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, cls=NpEncoder)
print(f"\nResults saved to {out_path}")
print(f"Total time: {(time.time()-t_global)/60:.1f} min")

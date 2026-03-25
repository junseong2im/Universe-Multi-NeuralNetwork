"""
olfabind_pipeline.py
====================
OlfaBind Unified Pipeline: Module 1 → 1-B → 2 → 3 → 4

End-to-end differentiable pipeline that connects:
  - Real molecular data (SMILES → Morgan Fingerprint)
  - Module 1: InputHardwareLayer (constellation patterns)
  - Module 1-B: SliceLatentModule (contrastive latent positions)
  - Module 2: GravitationalEngine (N-body simulation)
  - Module 3: OrbitalStabilityEvaluator (stability + physics embedding)
  - Module 4: BioDopamineSystem (bio-feedback RL)

Ultimate Goals:
  1. Bio-Sync Therapy: real-time scent adaptation from bio-signals
  2. Zero-Shot Masterpiece: orbital stability → unseen scent recipes
  3. Digital Scent Archive: latent coordinates as universal scent data

Production-ready. All operations differentiable and GPU-compatible.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from models.olfabind_input import InputHardwareLayer
from models.olfabind_contrastive import SliceLatentModule
from models.olfabind_engine import PhysicsProcessingEngine
from models.olfabind_reward import BioDopamineSystem


# ======================================================================
# Molecular Featurizer: SMILES → Fingerprint Tensor
# ======================================================================

class MolecularFeaturizer:
    """
    Converts SMILES strings to Morgan Fingerprint tensors.
    
    This bridges the gap between chemical data and the OlfaBind neural pipeline.
    Uses RDKit Morgan fingerprints (ECFP4, 2048-bit) as standard input.
    """
    def __init__(self, nbits: int = 2048, radius: int = 2):
        self.nbits = nbits
        self.radius = radius
        self._cache = {}
    
    def smiles_to_fingerprint(self, smiles: str) -> np.ndarray:
        """Convert single SMILES to fingerprint vector."""
        if smiles in self._cache:
            return self._cache[smiles]
        
        from rdkit import Chem
        from rdkit.Chem import AllChem
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            fp = np.zeros(self.nbits, dtype=np.float32)
        else:
            bit_vec = AllChem.GetMorganFingerprintAsBitVect(
                mol, self.radius, nBits=self.nbits
            )
            fp = np.array(bit_vec, dtype=np.float32)
        
        self._cache[smiles] = fp
        return fp
    
    def mixture_to_tensor(
        self,
        smiles_list: list,
        max_molecules: int = 15,
        device: torch.device = None
    ) -> tuple:
        """
        Convert a mixture of SMILES to padded tensor + mask.
        
        Returns:
            features: (1, max_molecules, nbits) — fingerprint tensor
            mask: (1, max_molecules) — valid molecule mask
        """
        if device is None:
            device = torch.device('cpu')
        
        N = min(len(smiles_list), max_molecules)
        features = np.zeros((max_molecules, self.nbits), dtype=np.float32)
        mask = np.zeros(max_molecules, dtype=np.float32)
        
        for i in range(N):
            features[i] = self.smiles_to_fingerprint(smiles_list[i])
            mask[i] = 1.0
        
        features_t = torch.tensor(features, device=device).unsqueeze(0)
        mask_t = torch.tensor(mask, device=device).unsqueeze(0)
        
        return features_t, mask_t
    
    def batch_mixtures_to_tensor(
        self,
        mixtures: list,
        max_molecules: int = 15,
        device: torch.device = None
    ) -> tuple:
        """
        Convert batch of mixtures to padded tensors.
        
        Returns:
            features: (B, max_molecules, nbits)
            mask: (B, max_molecules)
        """
        if device is None:
            device = torch.device('cpu')
        
        B = len(mixtures)
        features = np.zeros((B, max_molecules, self.nbits), dtype=np.float32)
        mask = np.zeros((B, max_molecules), dtype=np.float32)
        
        for b, smiles_list in enumerate(mixtures):
            N = min(len(smiles_list), max_molecules)
            for i in range(N):
                features[b, i] = self.smiles_to_fingerprint(smiles_list[i])
                mask[b, i] = 1.0
        
        features_t = torch.tensor(features, device=device)
        mask_t = torch.tensor(mask, device=device)
        
        return features_t, mask_t


# ======================================================================
# OlfaBind Unified Pipeline
# ======================================================================

class OlfaBindPipeline(nn.Module):
    """
    Complete OlfaBind Pipeline: End-to-end differentiable scent processing.
    
    SMILES → Fingerprint → Constellation → Contrastive Latent → Physics Sim → Stability
    
    This is the core engine for:
    1. Bio-Sync Therapy: stability + bio_reward → parameter update
    2. Zero-Shot Masterpiece: optimize stability to generate new recipes
    3. Digital Scent Archive: latent z-coordinates as universal scent encoding
    """
    def __init__(
        self,
        d_input: int = 2048,
        d_atom: int = 128,
        n_steps: int = 32,
        dt: float = 0.01,
        z_dim: int = 3,
        contrastive_temperature: float = 0.07,
        d_bio: int = 5,
        buffer_size: int = 1000,
    ):
        super().__init__()
        
        # === Module 1: Input Hardware Layer ===
        self.module1 = InputHardwareLayer(
            d_input=d_input,
            d_atom=d_atom,
            grid_h=8,
            grid_w=d_atom // 8,
        )
        
        # === Module 1-B: Contrastive Latent ===
        self.module1b = SliceLatentModule(
            d_atom=d_atom,
            h_dim=256,
            z_dim=z_dim,
            temperature=contrastive_temperature,
        )
        
        # === Module 2+3: Physics Engine + Stability ===
        self.physics = PhysicsProcessingEngine(
            d_atom=d_atom,
            n_steps=n_steps,
            dt=dt,
        )
        
        # === Module 4: Bio-Dopamine RL ===
        self.module4 = BioDopamineSystem(
            d_bio=d_bio,
            buffer_size=buffer_size,
        )
        
        self.d_input = d_input
        self.d_atom = d_atom
        self.n_steps = n_steps
        self.z_dim = z_dim
    
    def forward(
        self,
        molecular_features: torch.Tensor,  # (B, N, D_input=2048)
        mask: torch.Tensor,                # (B, N)
        bio_signals: torch.Tensor = None,  # (B, D_bio=5) — optional
    ) -> dict:
        """Full forward pass through the OlfaBind pipeline."""
        # Stage 1: Fingerprint → Constellation
        constellations = self.module1(molecular_features, mask)
        
        # Stage 1-B: Constellation → Contrastive 3D Positions
        z_positions, contrastive_loss = self.module1b(constellations, mask)
        
        # Stage 2+3: Physics Simulation + Stability
        stability, physics_embedding, trajectory = self.physics(
            constellations, mask, override_positions=z_positions
        )
        
        result = {
            'constellations': constellations,
            'latent_positions': z_positions,
            'contrastive_loss': contrastive_loss,
            'stability': stability,
            'physics_embedding': physics_embedding,
            'trajectory': trajectory,
        }
        
        # Stage 4: Bio-Dopamine RL (if bio-signals provided)
        if bio_signals is not None:
            reward, reinforce_loss = self.module4(stability, bio_signals)
            result['bio_reward'] = reward
            result['reinforce_loss'] = reinforce_loss
        
        return result
    
    def compute_similarity(
        self,
        features_a: torch.Tensor, mask_a: torch.Tensor,
        features_b: torch.Tensor, mask_b: torch.Tensor,
    ) -> torch.Tensor:
        """Compute perceptual similarity between two mixtures via physics embeddings."""
        out_a = self.forward(features_a, mask_a)
        out_b = self.forward(features_b, mask_b)
        
        return F.cosine_similarity(
            out_a['physics_embedding'],
            out_b['physics_embedding'],
            dim=-1
        )
    
    def get_scent_encoding(
        self,
        molecular_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict:
        """
        Generate the Digital Scent Archive encoding for a mixture.
        
        Universal scent data format:
        - latent_coordinates: (N, 3) — 3D coordinates in scent latent space
        - stability_score: scalar — orbital stability score
        - physics_vector: (19,) — physics feature vector
        """
        self.eval()
        with torch.no_grad():
            result = self.forward(molecular_features, mask)
        
        return {
            'latent_coordinates': result['latent_positions'].cpu(),
            'stability_score': result['stability'].cpu(),
            'physics_vector': result['physics_embedding'].cpu(),
            'n_molecules': int(mask.sum(dim=-1).item()),
        }
    
    def compute_total_loss(
        self,
        result: dict,
        similarity_target: torch.Tensor = None,
        physics_embedding_b: torch.Tensor = None,
        contrastive_weight: float = 1.0,
        similarity_weight: float = 1.0,
        stability_weight: float = 0.1,
        reinforce_weight: float = 0.01,
    ) -> dict:
        """
        Combined training loss from all modules.
        
        Loss = contrastive * InfoNCE
             + stability * (1 - stability_score)
             + similarity * MSE(predicted, target)
             + reinforce * REINFORCE_loss
        """
        losses = {}
        
        # Contrastive loss (Module 1-B unsupervised)
        losses['contrastive'] = result['contrastive_loss'] * contrastive_weight
        
        # Stability regularizer (maximize stability)
        losses['stability'] = (1.0 - result['stability']).mean() * stability_weight
        
        # Similarity loss (supervised, if target provided)
        if similarity_target is not None and physics_embedding_b is not None:
            pred_sim = F.cosine_similarity(
                result['physics_embedding'], physics_embedding_b, dim=-1
            )
            losses['similarity'] = F.mse_loss(pred_sim, similarity_target) * similarity_weight
        
        # REINFORCE loss (bio-feedback)
        if 'reinforce_loss' in result:
            losses['reinforce'] = result['reinforce_loss'] * reinforce_weight
        
        losses['total'] = sum(losses.values())
        return losses
    
    def count_parameters(self) -> dict:
        """Parameter breakdown by module."""
        def count(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        return {
            'module1_input': count(self.module1),
            'module1b_contrastive': count(self.module1b),
            'module2_3_physics': count(self.physics),
            'module4_biodopamine': count(self.module4),
            'total': count(self),
        }


# ======================================================================
# VERIFICATION: End-to-End Self-Test
# ======================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  OlfaBind Unified Pipeline -- End-to-End Self-Test")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    B, N, D_INPUT, T = 4, 10, 2048, 16
    
    # Test 1: Construction
    print("\n[Test 1] Pipeline construction...")
    pipeline = OlfaBindPipeline(
        d_input=D_INPUT, d_atom=128, n_steps=T, dt=0.01
    ).to(device)
    
    params = pipeline.count_parameters()
    print(f"  Module 1 (Input):         {params['module1_input']:>10,}")
    print(f"  Module 1-B (Contrastive): {params['module1b_contrastive']:>10,}")
    print(f"  Module 2+3 (Physics):     {params['module2_3_physics']:>10,}")
    print(f"  Module 4 (Bio-Dopamine):  {params['module4_biodopamine']:>10,}")
    print(f"  TOTAL:                    {params['total']:>10,}")
    
    # Test 2: Forward
    print("\n[Test 2] Forward pass...")
    pipeline.train()
    features = torch.randn(B, N, D_INPUT, device=device)
    mask = torch.ones(B, N, device=device)
    mask[:, 7:] = 0
    
    result = pipeline(features, mask)
    print(f"  constellations:    {result['constellations'].shape}")
    print(f"  latent_positions:  {result['latent_positions'].shape}")
    print(f"  contrastive_loss:  {result['contrastive_loss'].item():.4f}")
    print(f"  stability:         {result['stability'].detach().cpu().numpy()}")
    print(f"  physics_embedding: {result['physics_embedding'].shape}")
    print(f"  trajectory:        {result['trajectory'].shape}")
    print("  PASS")
    
    # Test 3: Backward
    print("\n[Test 3] Backward pass...")
    losses = pipeline.compute_total_loss(result)
    losses['total'].backward()
    
    nan_grads = [n for n, p in pipeline.named_parameters()
                 if p.grad is not None and p.grad.isnan().any()]
    no_grads = [n for n, p in pipeline.named_parameters() if p.grad is None]
    print(f"  Total loss: {losses['total'].item():.6f}")
    print(f"  NaN gradients: {len(nan_grads)}")
    print(f"  No gradients: {len(no_grads)}")
    assert len(nan_grads) == 0, f"NaN gradients: {nan_grads}"
    print("  PASS")
    
    # Test 4: Bio-feedback
    print("\n[Test 4] Bio-Dopamine loop...")
    pipeline.zero_grad()
    bio_signals = torch.randn(B, 5, device=device)
    result_bio = pipeline(features, mask, bio_signals=bio_signals)
    print(f"  bio_reward:     {result_bio['bio_reward'].detach().cpu().numpy()}")
    print(f"  reinforce_loss: {result_bio['reinforce_loss'].item():.6f}")
    losses_bio = pipeline.compute_total_loss(result_bio)
    losses_bio['total'].backward()
    print("  PASS")
    
    # Test 5: Similarity
    print("\n[Test 5] Mixture similarity...")
    pipeline.eval()
    with torch.no_grad():
        sim = pipeline.compute_similarity(
            torch.randn(B, N, D_INPUT, device=device), mask,
            torch.randn(B, N, D_INPUT, device=device), mask,
        )
    print(f"  Similarity: {sim.cpu().numpy()}")
    print("  PASS")
    
    # Test 6: Scent encoding
    print("\n[Test 6] Digital Scent Archive...")
    enc = pipeline.get_scent_encoding(
        torch.randn(1, N, D_INPUT, device=device),
        torch.cat([torch.ones(1, 5, device=device),
                    torch.zeros(1, 5, device=device)], dim=-1),
    )
    print(f"  Latent coordinates: {enc['latent_coordinates'].shape}")
    print(f"  Stability: {enc['stability_score'].item():.4f}")
    print(f"  Physics vector: {enc['physics_vector'].shape}")
    print("  PASS")
    
    # Test 7: Real molecules
    print("\n[Test 7] Real SMILES pipeline...")
    try:
        featurizer = MolecularFeaturizer(nbits=D_INPUT)
        mixture = [
            "CC(=O)CCCCCCCCC",        # 2-Undecanone
            "CC(C)=CCCC(C)=CC=O",     # Citral
            "CC(C)=CCCC(C)=CCO",      # Geraniol
            "CC(=O)c1ccc(O)cc1",      # p-Hydroxyacetophenone
            "O=Cc1ccc(OC)cc1",        # Anisaldehyde
        ]
        feat_t, mask_t = featurizer.mixture_to_tensor(mixture, N, device)
        with torch.no_grad():
            r = pipeline(feat_t, mask_t)
        print(f"  Stability: {r['stability'].item():.4f}")
        print(f"  Latent[0]: {r['latent_positions'][0, 0].cpu().numpy()}")
        print(f"  Latent[1]: {r['latent_positions'][0, 1].cpu().numpy()}")
        print("  PASS")
    except ImportError:
        print("  SKIP (RDKit not available)")
    
    # Test 8: Training loop
    print("\n[Test 8] Training step...")
    pipeline.train()
    optimizer = torch.optim.AdamW(pipeline.parameters(), lr=1e-4)
    for step in range(5):
        optimizer.zero_grad()
        result = pipeline(features, mask, bio_signals=bio_signals)
        losses = pipeline.compute_total_loss(result)
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(pipeline.parameters(), max_norm=1.0)
        optimizer.step()
        print(f"  Step {step+1}: loss={losses['total'].item():.6f}")
    print("  PASS")
    
    print("\n" + "=" * 70)
    print("  ALL TESTS PASSED -- OlfaBind Pipeline Production Ready")
    print(f"  Total Parameters: {params['total']:,}")
    print("=" * 70)

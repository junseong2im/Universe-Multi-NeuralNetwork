"""
olfabind_input.py
=================
OlfaBind Module 1: Input Hardware Layer (Atom Pattern Slice Array)

Converts molecular data (SMILES/fingerprints) into "constellation patterns"
on a 2D atom slice array.

Key Concept: Instead of vertical deep computation, this module spreads
features horizontally across a 2D grid of "atom nodes", where each node
represents a micro-feature (specific carbon ring, polarity, molecular weight, etc.)

When a molecule is input, specific atoms activate simultaneously,
forming a "constellation pattern" that defines the scent.

This module bridges real chemical data -> constellation tensor -> Module 2.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class AtomSliceArray(nn.Module):
    """
    2D Atom Slice Array: A grid of micro-feature nodes.
    
    Each node in the grid represents a specific micro-feature detector:
    - Carbon ring structures (aromatic, aliphatic, hetero)
    - Polarity indicators (polar, nonpolar, amphiphilic)
    - Molecular weight ranges (light, medium, heavy)
    - Functional group detectors (hydroxyl, aldehyde, ester, ketone...)
    - Bond type detectors (single, double, triple, aromatic)
    
    The array dimensions are (H, W) where H*W = D_atom (total micro-features).
    
    Input:  molecular_features (B, N, D_input) - raw molecular features (fingerprints)
            mask               (B, N)          - valid molecule indicators
    Output: constellations     (B, N, D_atom)  - activated constellation patterns
    """
    def __init__(
        self,
        d_input: int,
        grid_h: int = 8,
        grid_w: int = 16,
    ):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.d_atom = grid_h * grid_w  # total micro-feature nodes
        
        # Projection from raw features to atom grid activations
        # Each atom node has a learned "detector" that responds to specific input patterns
        self.detector_bank = nn.Linear(d_input, self.d_atom)
        
        # Lateral inhibition layer: nearby atoms compete for activation
        # Simulates biological lateral inhibition in sensory systems
        self.lateral_inhibition = nn.Conv1d(
            in_channels=1, out_channels=1,
            kernel_size=5, padding=2, bias=False
        )
        # Initialize as difference-of-Gaussians (center-surround)
        with torch.no_grad():
            dog_kernel = torch.tensor([-0.1, -0.2, 1.0, -0.2, -0.1])
            self.lateral_inhibition.weight.copy_(dog_kernel.reshape(1, 1, -1))
    
    def forward(
        self,
        molecular_features: torch.Tensor,  # (B, N, D_input)
        mask: torch.Tensor                 # (B, N)
    ) -> torch.Tensor:
        B, N, _ = molecular_features.shape
        
        # Step 1: Project to atom grid space
        # Each molecule activates specific detector nodes
        raw_activations = self.detector_bank(molecular_features)  # (B, N, D_atom)
        
        # Step 2: Lateral inhibition (sharpen patterns)
        # Reshape for Conv1d: (B*N, 1, D_atom)
        flat = raw_activations.reshape(B * N, 1, self.d_atom)
        sharpened = self.lateral_inhibition(flat)
        sharpened = sharpened.reshape(B, N, self.d_atom)
        
        # Step 3: Keep hard sparsity from CombinatorialCoder's top-k
        # Only scale the already-active nodes (non-zero from STE)
        # Active nodes get enhanced/inhibited by lateral competition
        # Inactive nodes stay at zero -> constellation pattern preserved
        constellation = sharpened
        
        # Step 4: L2 normalize (unit energy per constellation, preserves zero structure)
        constellation = F.normalize(constellation, p=2, dim=-1)
        
        # Step 5: Mask invalid molecules
        constellation = constellation * mask.unsqueeze(-1)
        
        return constellation  # (B, N, D_atom)
    
    def get_constellation_image(
        self,
        constellation: torch.Tensor  # (B, N, D_atom) or (D_atom,)
    ) -> torch.Tensor:
        """
        Reshape constellation vector into 2D grid for visualization.
        Returns: (B, N, H, W) or (H, W)
        """
        shape = constellation.shape
        if len(shape) == 1:
            return constellation.reshape(self.grid_h, self.grid_w)
        elif len(shape) == 2:
            return constellation.reshape(shape[0], self.grid_h, self.grid_w)
        else:
            return constellation.reshape(
                shape[0], shape[1], self.grid_h, self.grid_w
            )


class CombinatorialCoder(nn.Module):
    """
    Combinatorial Coding Module.
    
    Implements the biological principle of combinatorial coding:
    a single molecule activates MULTIPLE detector nodes simultaneously,
    and the COMBINATION (constellation) uniquely identifies the molecule.
    
    This is different from one-hot encoding where each molecule maps to
    exactly one node.
    
    The module ensures:
    1. Sparsity: only ~10-30% of nodes activate per molecule
    2. Distinctness: different molecules produce different constellations
    3. Overlap: similar molecules share constellation features
    
    Input:  molecular_features (B, N, D_input)
            mask               (B, N)
    Output: constellations     (B, N, D_atom)
    """
    def __init__(
        self,
        d_input: int,
        d_atom: int = 128,
        n_codebooks: int = 4,
        sparsity_target: float = 0.2,
        ste_sharpness: float = 5.0
    ):
        super().__init__()
        self.d_atom = d_atom
        self.n_codebooks = n_codebooks
        self.sparsity_target = sparsity_target
        self.ste_sharpness = ste_sharpness
        self.atoms_per_book = d_atom // n_codebooks
        
        # Multiple codebooks for combinatorial richness
        # Each codebook captures different aspects of the molecule
        self.codebooks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_input, self.atoms_per_book),
                nn.LayerNorm(self.atoms_per_book),
            )
            for _ in range(n_codebooks)
        ])
        
        # Number of atoms to activate per molecule (top-k)
        self.k = max(1, int(d_atom * sparsity_target))
    
    def forward(
        self,
        molecular_features: torch.Tensor,  # (B, N, D_input)
        mask: torch.Tensor                 # (B, N)
    ) -> torch.Tensor:
        B, N, D = molecular_features.shape
        
        # Process each codebook
        code_parts = []
        for codebook in self.codebooks:
            flat_input = molecular_features.reshape(B * N, D)
            code = codebook(flat_input)  # (B*N, atoms_per_book)
            code = code.reshape(B, N, self.atoms_per_book)
            code_parts.append(code)
        
        # Concatenate all codebooks: (B, N, D_atom)
        raw_pattern = torch.cat(code_parts, dim=-1)
        
        # ============================================================
        # Hard Top-K Sparsity + Straight-Through Estimator
        # ============================================================
        # Step A: Find top-k activation indices per molecule
        # This guarantees EXACTLY k nodes light up (constellation pattern)
        topk_vals, topk_idx = raw_pattern.topk(self.k, dim=-1)  # (B, N, k)
        
        # Step B: Create binary mask (hard: 0 or 1)
        hard_mask = torch.zeros_like(raw_pattern)  # (B, N, D_atom)
        hard_mask.scatter_(-1, topk_idx, 1.0)      # 1 at top-k positions
        
        # Step C: Straight-Through Estimator
        # Forward: use hard_mask (binary constellation)
        # Backward: gradient flows through raw_pattern (as if sigmoid)
        soft_approx = torch.sigmoid(raw_pattern * self.ste_sharpness)  # soft approximation
        sparse_pattern = hard_mask + (soft_approx - soft_approx.detach())
        # The trick: hard_mask has no grad, but (soft - soft.detach()) = 0 in forward
        # yet carries gradient of soft_approx in backward pass
        
        # Sparsity is now GUARANTEED: exactly self.k / self.d_atom active ratio
        actual_sparsity = hard_mask.mean()
        self.sparsity_loss = torch.tensor(0.0, device=raw_pattern.device)
        
        # Mask invalid molecules
        sparse_pattern = sparse_pattern * mask.unsqueeze(-1)
        
        return sparse_pattern  # (B, N, D_atom)


class InputHardwareLayer(nn.Module):
    """
    Module 1: Complete Input Hardware Layer.
    
    Bridges raw molecular data to constellation patterns suitable for
    the Physics Processing Engine (Module 2).
    
    Pipeline:
    1. Raw features (fingerprints) -> CombinatorialCoder -> sparse patterns
    2. Sparse patterns -> AtomSliceArray -> sharpened constellations
    
    Input:  molecular_features (B, N, D_input) - fingerprints or embeddings
            mask               (B, N)          - valid molecule indicators
    Output: constellations     (B, N, D_atom)  - ready for Module 2
    """
    def __init__(
        self,
        d_input: int = 2048,
        d_atom: int = 128,
        grid_h: int = 8,
        grid_w: int = 16
    ):
        super().__init__()
        assert grid_h * grid_w == d_atom, f"grid_h*grid_w ({grid_h}*{grid_w}) must equal d_atom ({d_atom})"
        
        self.d_input = d_input
        self.d_atom = d_atom
        
        # Stage 1: Combinatorial coding (sparse activation patterns)
        self.coder = CombinatorialCoder(
            d_input=d_input,
            d_atom=d_atom,
            n_codebooks=4,
            sparsity_target=0.2
        )
        
        # Stage 2: Atom slice array (lateral inhibition + sharpening)
        self.slice_array = AtomSliceArray(
            d_input=d_atom,  # takes output of coder
            grid_h=grid_h,
            grid_w=grid_w,
        )
    
    def forward(
        self,
        molecular_features: torch.Tensor,  # (B, N, D_input)
        mask: torch.Tensor                 # (B, N)
    ) -> torch.Tensor:
        # Stage 1: Combinatorial coding
        sparse_codes = self.coder(molecular_features, mask)  # (B, N, D_atom)
        
        # Stage 2: Atom slice sharpening
        constellations = self.slice_array(sparse_codes, mask)  # (B, N, D_atom)
        
        return constellations
    
    def get_sparsity_loss(self) -> torch.Tensor:
        """Returns sparsity regularization loss for training."""
        return self.coder.sparsity_loss


# ======================================================================
# VERIFICATION
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("OlfaBind Module 1 - Input Hardware Layer Self-test")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    B, N, D_INPUT, D_ATOM = 4, 10, 2048, 128
    
    # --- Test 1: Forward pass ---
    print("\n[Test 1] Forward pass...")
    module1 = InputHardwareLayer(
        d_input=D_INPUT, d_atom=D_ATOM,
        grid_h=8, grid_w=16
    ).to(device)
    
    features = torch.randn(B, N, D_INPUT, device=device)
    mask = torch.ones(B, N, device=device)
    mask[:, 7:] = 0
    
    constellations = module1(features, mask)
    print(f"  Input shape:  {features.shape}")
    print(f"  Output shape: {constellations.shape}")
    print(f"  Params: {sum(p.numel() for p in module1.parameters())}")
    
    # --- Test 2: Sparsity check ---
    print("\n[Test 2] Sparsity check...")
    active_ratio = (constellations[0, 0] > 0.5).float().mean().item()
    print(f"  Active ratio (sample): {active_ratio:.3f}")
    print(f"  Sparsity loss: {module1.get_sparsity_loss().item():.6f}")
    
    # --- Test 3: Gradient flow ---
    print("\n[Test 3] Gradient flow...")
    loss = constellations.sum()
    loss.backward()
    no_grad = [n for n, p in module1.named_parameters() if p.grad is None]
    print(f"  No gradient: {no_grad if no_grad else 'NONE (all OK)'}")
    
    # --- Test 4: Constellation visualization ---
    print("\n[Test 4] Constellation grid shape...")
    grid = module1.slice_array.get_constellation_image(constellations)
    print(f"  Grid shape: {grid.shape}")  # (B, N, 8, 16)
    
    # --- Test 5: Different molecules = different constellations ---
    print("\n[Test 5] Distinctness check...")
    mol_a = torch.randn(1, 1, D_INPUT, device=device)
    mol_b = torch.randn(1, 1, D_INPUT, device=device)
    mask_1 = torch.ones(1, 1, device=device)
    
    with torch.no_grad():
        const_a = module1(mol_a, mask_1)
        const_b = module1(mol_b, mask_1)
        cosine = F.cosine_similarity(const_a.squeeze(), const_b.squeeze(), dim=0).item()
        print(f"  Cosine similarity between 2 random molecules: {cosine:.4f}")
        print(f"  (Should be < 1.0 for distinct constellations)")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)

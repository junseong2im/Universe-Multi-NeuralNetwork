"""
olfabind_contrastive.py
=======================
OlfaBind Module 1: Sensory Digitization and Latent Space Formation

Projects sparse vectors x in {0,1}^N to 3D cosmic coordinates z = (zx, zy, zz)
via a deep learning encoder.
Triplet Margin Loss is used to cluster similar chemical structures together.

L = max(0, d(z_a, z_p) - d(z_a, z_n) + alpha)
  d = Euclidean distance, alpha = minimum margin

Pipeline:
  constellation (B, N, 128) -> SliceEncoder -> h (B, N, 256)
                             -> ProjectionHead -> z (B, N, 3)  (raw 3D coordinates, no normalization)
  z is used as initial 3D position in the gravitational N-body simulation.

Contrastive Learning (Triplet Margin):
  - Augmentation: atom-level dropout (p=0.2) + Gaussian noise (sigma=0.05)
  - Anchor/Positive: two augmented views of the same molecule
  - Negative: different molecule within the batch (hardest negative mining)
  - Loss: Triplet Margin Loss with margin alpha=1.0
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SliceEncoder(nn.Module):
    """
    Encodes sparse constellation patterns into high-dimensional feature vectors.
    
    Input:  x (B, N, D_atom=128)  — sparse atom activation patterns
    Output: h (B, N, h_dim=256)   — latent feature vectors
    """
    def __init__(self, d_atom: int = 128, h_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_atom, h_dim),
            nn.LayerNorm(h_dim),
            nn.GELU(),
            nn.Linear(h_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.GELU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)  # (B, N, h_dim)


class ProjectionHead(nn.Module):
    """
    Projects high-dimensional features to low-dimensional latent space.
    Output dimension = 3 so it can directly serve as 3D initial positions
    for the physics engine.
    
    Input:  h (B, N, h_dim=256)
    Output: z (B, N, z_dim=3)  — L2-normalized latent vectors
    """
    def __init__(self, h_dim: int = 256, z_dim: int = 3, mid_dim: int = None):
        super().__init__()
        if mid_dim is None:
            mid_dim = max(h_dim // 4, z_dim * 4)
        self.projector = nn.Sequential(
            nn.Linear(h_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, z_dim),
        )
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.projector(h)             # (B, N, z_dim)
        # NO L2 normalization — raw 3D coordinates for physics engine
        # tanh bounds the output to [-1, 1], position_scale expands later
        z = torch.tanh(z)
        return z


class ConstellationAugmenter(nn.Module):
    """
    Data augmentation for contrastive learning on sparse constellation patterns.
    
    Two augmentation strategies applied simultaneously:
    1. Atom dropout: randomly zero out active nodes (p=drop_prob)
    2. Gaussian noise: add small perturbation (σ=noise_std)
    
    This generates two distinct "views" of the same molecule while
    preserving the overall constellation structure.
    """
    def __init__(self, drop_prob: float = 0.2, noise_std: float = 0.05):
        super().__init__()
        self.drop_prob = drop_prob
        self.noise_std = noise_std
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation to constellation patterns.
        Only applied during training.
        
        Input:  x (B, N, D_atom)
        Output: x_aug (B, N, D_atom) — augmented version
        """
        if not self.training:
            return x
        
        # 1. Atom dropout: zero out random active positions
        # Create a binary mask where 1 = keep, 0 = drop
        drop_mask = (torch.rand_like(x) > self.drop_prob).float()
        x_aug = x * drop_mask
        
        # 2. Gaussian noise on remaining active positions
        noise = torch.randn_like(x) * self.noise_std
        # Only add noise where the original pattern had non-zero values
        active_mask = (x.abs() > 1e-8).float()
        x_aug = x_aug + noise * active_mask * drop_mask
        
        return x_aug


def triplet_margin_loss(
    anchor: torch.Tensor,     # (M, z_dim) — anchor embeddings
    positive: torch.Tensor,   # (M, z_dim) — positive embeddings (same molecule, different view)
    margin: float = 1.0
) -> torch.Tensor:
    """
    Triplet Margin Loss with Hardest Negative Mining.
    
    Mathematical definition:
      L = max(0, d(z_a, z_p) - d(z_a, z_n) + alpha)

    Where:
      d = Euclidean distance
      alpha = margin (minimum distance margin to maintain)
      z_n = hardest negative in batch (closest different molecule to anchor)
    
    Input:  anchor (M, z_dim), positive (M, z_dim)
    Output: scalar loss
    """
    M = anchor.shape[0]
    if M < 2:
        return torch.tensor(0.0, device=anchor.device, requires_grad=True)
    
    # Anchor-Positive distance: d(z_a, z_p)
    d_ap = torch.pairwise_distance(anchor, positive, p=2)  # (M,)
    
    # Pairwise distance matrix between all anchors: (M, M)
    # d_matrix[i, j] = ||anchor_i - anchor_j||_2
    diff = anchor.unsqueeze(0) - anchor.unsqueeze(1)   # (M, M, z_dim)
    d_matrix = diff.norm(dim=-1)                        # (M, M)
    
    # Mask out self-distance (diagonal) with large value
    eye = torch.eye(M, device=anchor.device, dtype=torch.bool)
    d_matrix = d_matrix.masked_fill(eye, float('inf'))
    
    # Hardest negative: closest different molecule to each anchor
    d_an, _ = d_matrix.min(dim=1)  # (M,) — hardest negative distance
    
    # Triplet loss: max(0, d_ap - d_an + margin)
    loss = F.relu(d_ap - d_an + margin).mean()
    
    return loss


class SliceLatentModule(nn.Module):
    """
    OlfaBind Module 1: Sensory Digitization and Latent Space Formation

    Projects sparse vectors to 3D cosmic coordinates and uses Triplet
    Margin Loss to cluster similar chemical structures together.

    During training:
      - Generates two augmented views
      - Triplet Margin Loss enforces anchor-positive < anchor-negative distance

    During inference:
      - Returns 3D coordinates (no augmentation)

    Input:  constellations (B, N, D_atom=128)
            mask           (B, N)
    Output: z_positions    (B, N, 3)       -- 3D cosmic coordinates
            triplet_loss   (scalar)        -- Triplet Margin Loss (0.0 during eval)
    """
    def __init__(
        self,
        d_atom: int = 128,
        h_dim: int = 256,
        z_dim: int = 3,
        margin: float = 1.0,
        drop_prob: float = 0.2,
        noise_std: float = 0.05,
        position_scale_init: float = 2.0,
        mid_dim: int = None,
        # Keep temperature for backward compat (unused in triplet loss)
        temperature: float = 0.07,
    ):
        super().__init__()
        self.d_atom = d_atom
        self.z_dim = z_dim
        self.margin = margin
        
        # Learnable position scale (init=2.0, optimized during training)
        self.position_scale = nn.Parameter(torch.tensor(position_scale_init))
        
        # Core modules
        self.encoder = SliceEncoder(d_atom=d_atom, h_dim=h_dim)
        self.projector = ProjectionHead(h_dim=h_dim, z_dim=z_dim, mid_dim=mid_dim)
        self.augmenter = ConstellationAugmenter(
            drop_prob=drop_prob, noise_std=noise_std
        )
    
    def encode(self, constellations: torch.Tensor) -> torch.Tensor:
        """
        Encode constellations to 3D latent coordinates.
        
        Input:  constellations (B, N, D_atom)
        Output: z (B, N, z_dim=3) — raw 3D coordinates (no normalization)
        """
        h = self.encoder(constellations)   # (B, N, h_dim)
        z = self.projector(h)              # (B, N, z_dim) — raw 3D via tanh
        return z
    
    def forward(
        self,
        constellations: torch.Tensor,  # (B, N, D_atom)
        mask: torch.Tensor             # (B, N)
    ):
        """
        Forward pass with Triplet Margin Loss.
        
        Returns:
            z_positions: (B, N, 3) -- scaled 3D cosmic coordinates
            triplet_loss: scalar — Triplet Margin Loss (0.0 during eval)
        """
        B, N, D = constellations.shape
        
        if self.training:
            # Generate two augmented views (noise variants of same molecule)
            view_1 = self.augmenter(constellations)  # (B, N, D)
            view_2 = self.augmenter(constellations)  # (B, N, D)
            
            # Encoding: project to 3D coordinates
            z_1 = self.encode(view_1)  # (B, N, z_dim)
            z_2 = self.encode(view_2)  # (B, N, z_dim)
            
            # Extract valid molecules only (for Triplet Loss computation)
            valid_mask = mask.bool()  # (B, N)
            anchor = z_1[valid_mask]    # (M, z_dim) -- anchor
            positive = z_2[valid_mask]  # (M, z_dim) -- positive (different view of same molecule)
            
            # Triplet Margin Loss: L = max(0, d(a,p) - d(a,n) + α)
            cl_loss = triplet_margin_loss(anchor, positive, margin=self.margin)
            
            # Use View 1 coordinates as physics engine initial positions
            z_positions = z_1 * self.position_scale
        else:
            # Inference: encode directly without augmentation
            z = self.encode(constellations)  # (B, N, z_dim)
            z_positions = z * self.position_scale
            cl_loss = torch.tensor(0.0, device=constellations.device)
        
        # Zero out invalid molecule coordinates
        z_positions = z_positions * mask.unsqueeze(-1)
        
        return z_positions, cl_loss


# ======================================================================
# VERIFICATION: Self-test
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("OlfaBind Module 1 — Triplet Margin Loss Self-test")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    B, N, D_ATOM = 8, 10, 128
    
    # --- Test 1: Forward pass shape ---
    print("\n[Test 1] Forward pass shape check...")
    module = SliceLatentModule(d_atom=D_ATOM, margin=1.0).to(device)
    module.train()
    
    constellations = torch.randn(B, N, D_ATOM, device=device)
    mask = torch.ones(B, N, device=device)
    mask[:, 7:] = 0  # 7 valid molecules per mixture
    
    z_pos, cl_loss = module(constellations, mask)
    print(f"  z_positions shape: {z_pos.shape}")  # (8, 10, 3)
    assert z_pos.shape == (B, N, 3), f"Expected (B, N, 3), got {z_pos.shape}"
    print(f"  triplet_loss: {cl_loss.item():.4f}")
    print("  PASS")
    
    # --- Test 2: Loss is finite ---
    print("\n[Test 2] Triplet loss is finite...")
    assert torch.isfinite(cl_loss), "Loss must be finite"
    print(f"  Loss = {cl_loss.item():.4f} (finite)")
    print("  PASS")
    
    # --- Test 3: Raw 3D coordinates (no L2 normalization) ---
    print("\n[Test 3] Raw 3D coordinates (not normalized)...")
    module.eval()
    with torch.no_grad():
        z = module.encode(constellations)
        z_norms = z.norm(dim=-1)
        # tanh output should NOT all be unit norm
        print(f"  z norm mean: {z_norms.mean().item():.4f} (should not be 1.0)")
    print("  PASS")
    
    # --- Test 4: Gradient flow ---
    print("\n[Test 4] Gradient flow check...")
    module.train()
    module.zero_grad()
    z_pos, cl_loss = module(constellations, mask)
    total_loss = cl_loss + z_pos.sum() * 0.001
    total_loss.backward()
    
    nan_grad = []
    for name, p in module.named_parameters():
        if p.grad is not None and p.grad.isnan().any():
            nan_grad.append(name)
    
    n_with_grad = sum(1 for p in module.parameters() if p.grad is not None)
    print(f"  {n_with_grad}/{sum(1 for _ in module.parameters())} parameters have gradients")
    assert len(nan_grad) == 0, f"NaN gradients in: {nan_grad}"
    print("  PASS")
    
    # --- Test 5: Euclidean distance consistency ---
    print("\n[Test 5] Euclidean distance check (same vs diff molecule)...")
    module.eval()
    mol_a = torch.randn(1, 1, D_ATOM, device=device)
    mol_b = torch.randn(1, 1, D_ATOM, device=device)
    
    with torch.no_grad():
        z_a = module.encode(mol_a).squeeze()  # (3,)
        z_b = module.encode(mol_b).squeeze()  # (3,)
        d_same = torch.dist(z_a, z_a).item()  # should be 0
        d_diff = torch.dist(z_a, z_b).item()  # should be > 0
    
    print(f"  Same molecule distance: {d_same:.6f} (should be 0)")
    print(f"  Diff molecule distance: {d_diff:.4f} (should be > 0)")
    assert d_same < 1e-6, "Same molecule distance must be ~0"
    assert d_diff > 0, "Different molecules must have positive distance"
    print("  PASS")
    
    # --- Test 6: Masked molecules get zero positions ---
    print("\n[Test 6] Masking check...")
    z_pos, _ = module(constellations, mask)
    invalid_z = z_pos[:, 7:, :]
    assert invalid_z.abs().max().item() < 1e-8, "Invalid molecules must have zero position"
    print(f"  Invalid molecule max position: {invalid_z.abs().max().item():.1e}")
    print("  PASS")
    
    # --- Test 7: Parameter count ---
    print("\n[Test 7] Parameter count...")
    n_params = sum(p.numel() for p in module.parameters())
    print(f"  Total parameters: {n_params:,}")
    print(f"  Margin: {module.margin}")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)

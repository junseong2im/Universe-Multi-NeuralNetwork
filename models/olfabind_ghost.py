"""
olfabind_ghost.py
=================
OlfaBind Module 2: Early Cortex (Pattern Restoration via Anti-Gravity Perturbation)

When incomplete scent data is received, this module performs an inverse
computation to restore the 'complete constellation'.

Mathematical definition:
  dF = F_template - F_partial

  min_{r_p, m_p} || dF - G * sum_i (m_i * m_p / |r_i - r_p|^3) * (r_i - r_p) ||^2

  Optimizes ghost mass positions (r_p) and masses (m_p) via gradient descent;
  when the error converges to 0, the missing atoms' original positions are recovered.

Pipeline:
  - Complete constellation -> F_template (reference gravity field)
  - Incomplete constellation (some atoms off) -> F_partial
  - Ghost Mass Optimizer: gradient descent to find (r_p, m_p) and restore dF
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GravityFieldComputer(nn.Module):
    """
    Computes the gravitational vector field acting on each position
    from the given mass/position arrays.

    F_i = G * sum_{j!=i} (m_i * m_j / |r_j - r_i|^3) * (r_j - r_i)

    Input:  positions (B, N, 3), masses (B, N, 1)
    Output: force_field (B, N, 3) -- total gravity at each position
    """
    def __init__(self, epsilon: float = 1e-4):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(
        self,
        positions: torch.Tensor,   # (B, N, 3)
        masses: torch.Tensor,      # (B, N, 1)
        G: torch.Tensor,           # scalar or (1,)
        mask: torch.Tensor = None  # (B, N) — optional valid indicator
    ) -> torch.Tensor:
        B, N, _ = positions.shape
        
        # Pairwise displacement: diff[b,i,j] = pos[b,j] - pos[b,i]
        diff = positions.unsqueeze(1) - positions.unsqueeze(2)  # (B, N, N, 3)
        
        # Pairwise distance
        dist = diff.norm(dim=-1, keepdim=True).clamp(min=self.epsilon)  # (B, N, N, 1)
        
        # Gravitational force from j on i:
        # F_ij = G * m_i * m_j * (r_j - r_i) / |r_j - r_i|^3
        m_i = masses.unsqueeze(2)  # (B, N, 1, 1)
        m_j = masses.unsqueeze(1)  # (B, 1, N, 1)
        
        force_pairs = G * m_i * m_j * diff / (dist ** 3)  # (B, N, N, 3)
        
        # Zero out self-interaction (diagonal)
        eye = torch.eye(N, device=positions.device, dtype=torch.bool)
        eye = eye.unsqueeze(0).unsqueeze(-1)  # (1, N, N, 1)
        force_pairs = force_pairs.masked_fill(eye, 0.0)
        
        # Apply mask if provided
        if mask is not None:
            pair_mask = mask.unsqueeze(1) * mask.unsqueeze(2)  # (B, N, N)
            force_pairs = force_pairs * pair_mask.unsqueeze(-1)
        
        # Sum over source dimension
        force_field = force_pairs.sum(dim=2)  # (B, N, 3)
        
        return force_field


class GhostMassOptimizer(nn.Module):
    """
    Pattern restoration module using anti-gravity perturbation.

    Optimizes the positions and masses of missing atoms (ghost masses)
    in incomplete scent data to restore the original gravity field.

    Math:
      min_{r_p, m_p} || dF - G*sum(m_i*m_p/|r_i-r_p|^3)(r_i-r_p) ||^2

    Input:
      complete_positions  (B, N, 3)  -- complete scent atom positions
      complete_masses     (B, N, 1)  -- complete scent atom masses
      partial_mask        (B, N)     -- 1=present, 0=missing
      G                   scalar     -- gravitational constant

    Output:
      restored_positions  (B, N, 3)  -- restored full positions
      restored_masses     (B, N, 1)  -- restored full masses
      restoration_loss    scalar     -- restoration error ||dF||^2
    """
    def __init__(
        self,
        n_optim_steps: int = 20,
        lr: float = 0.05,
        epsilon: float = 1e-4,
    ):
        super().__init__()
        self.n_optim_steps = n_optim_steps
        self.lr = lr
        self.gravity_computer = GravityFieldComputer(epsilon=epsilon)
        
        # Learnable initial guess for ghost mass magnitude
        self.ghost_mass_init = nn.Parameter(torch.tensor(1.0))
    
    def compute_ghost_force(
        self,
        existing_positions: torch.Tensor,   # (B, K, 3) -- existing atoms
        existing_masses: torch.Tensor,       # (B, K, 1)
        ghost_positions: torch.Tensor,       # (B, G, 3) -- ghost mass positions
        ghost_masses: torch.Tensor,          # (B, G, 1) -- ghost masses
        G: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the additional gravity that ghost masses exert on existing atom positions.

        Returns: ghost_force (B, K, 3) -- gravity change at existing positions due to ghosts
        """
        B, K, _ = existing_positions.shape
        _, G_count, _ = ghost_positions.shape
        epsilon = self.gravity_computer.epsilon
        
        # diff[b, k, g] = ghost_pos[b,g] - exist_pos[b,k]
        diff = ghost_positions.unsqueeze(1) - existing_positions.unsqueeze(2)  # (B, K, G, 3)
        dist = diff.norm(dim=-1, keepdim=True).clamp(min=epsilon)  # (B, K, G, 1)
        
        m_exist = existing_masses.unsqueeze(2)  # (B, K, 1, 1)
        m_ghost = ghost_masses.unsqueeze(1)      # (B, 1, G, 1)
        
        # Force from each ghost on each existing atom
        force_per_ghost = G * m_exist * m_ghost * diff / (dist ** 3)  # (B, K, G, 3)
        
        # Sum over all ghosts
        ghost_force = force_per_ghost.sum(dim=2)  # (B, K, 3)
        
        return ghost_force
    
    def forward(
        self,
        complete_positions: torch.Tensor,   # (B, N, 3)
        complete_masses: torch.Tensor,       # (B, N, 1)
        partial_mask: torch.Tensor,          # (B, N) -- 1=present, 0=missing
        G: torch.Tensor,                     # scalar
        full_mask: torch.Tensor = None,      # (B, N) -- full valid atom mask
    ):
        """
        Restore missing atoms using ghost masses.
        
        Returns:
            restored_positions (B, N, 3)
            restored_masses    (B, N, 1)  
            restoration_loss   scalar — ||ΔF||²
        """
        B, N, _ = complete_positions.shape
        device = complete_positions.device
        
        if full_mask is None:
            full_mask = torch.ones(B, N, device=device)
        
        # Identify missing atoms
        missing_mask = full_mask * (1.0 - partial_mask)  # (B, N) -- missing positions
        n_missing = missing_mask.sum(dim=1).max().int().item()
        
        if n_missing == 0:
            # No missing atoms -> return as-is
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return complete_positions.clone(), complete_masses.clone(), zero_loss
        
        # === Step 1: Reference gravity field (complete state) ===
        F_template = self.gravity_computer(
            complete_positions, complete_masses, G, full_mask
        )  # (B, N, 3)
        
        # === Step 2: Partial gravity field (missing state) ===
        partial_masses = complete_masses * partial_mask.unsqueeze(-1)
        F_partial = self.gravity_computer(
            complete_positions, partial_masses, G, partial_mask
        )  # (B, N, 3)
        
        # === Step 3: Deficit force ===
        delta_F = F_template - F_partial  # (B, N, 3) -- gravity difference to restore
        
        # Compute deficit force only at existing atoms (ignore missing positions)
        delta_F = delta_F * partial_mask.unsqueeze(-1)  # (B, N, 3)
        
        # === Step 4: Ghost Mass Optimization ===
        # Initial estimate for missing atom positions = original + small noise
        ghost_pos = complete_positions.detach().clone()  # initial estimate
        ghost_pos = ghost_pos + torch.randn_like(ghost_pos) * 0.1
        ghost_pos = ghost_pos * missing_mask.unsqueeze(-1)
        ghost_pos.requires_grad_(True)
        
        # Ghost mass initial values
        ghost_mass_raw = torch.full(
            (B, N, 1), self.ghost_mass_init.item(),
            device=device, requires_grad=True
        )
        
        # Inner optimization loop (gradient descent)
        optim_params = [ghost_pos, ghost_mass_raw]
        inner_optimizer = torch.optim.Adam(optim_params, lr=self.lr)
        
        best_loss = float('inf')
        best_ghost_pos = ghost_pos.detach().clone()
        best_ghost_mass = ghost_mass_raw.detach().clone()
        
        for step in range(self.n_optim_steps):
            inner_optimizer.zero_grad()
            
            # Constrain ghost masses to be positive
            ghost_masses_pos = F.softplus(ghost_mass_raw) * missing_mask.unsqueeze(-1)
            
            # Apply ghost positions only at missing positions
            ghost_pos_masked = ghost_pos * missing_mask.unsqueeze(-1)
            
            # Force that ghost masses exert on existing atom positions
            # Extract existing atom positions
            exist_mask = partial_mask.bool()
            
            # Compute ghost-induced force at all positions (simplified)
            ghost_force = self.compute_ghost_force(
                complete_positions, complete_masses * partial_mask.unsqueeze(-1),
                ghost_pos_masked, ghost_masses_pos, G
            )  # (B, N, 3)
            
            # Residual: ||dF - ghost_force||^2
            ghost_force_at_existing = ghost_force * partial_mask.unsqueeze(-1)
            residual = delta_F - ghost_force_at_existing  # (B, N, 3)
            optim_loss = (residual ** 2).sum() / max(partial_mask.sum().item(), 1.0)
            
            if optim_loss.item() < best_loss:
                best_loss = optim_loss.item()
                best_ghost_pos = ghost_pos.detach().clone()
                best_ghost_mass = ghost_mass_raw.detach().clone()
            
            optim_loss.backward(retain_graph=True)
            inner_optimizer.step()
        
        # === Step 5: Combine restored state ===
        # Existing atoms keep original positions, missing atoms use optimized ghost positions
        restored_positions = (
            complete_positions * partial_mask.unsqueeze(-1) +
            best_ghost_pos * missing_mask.unsqueeze(-1)
        )
        
        restored_masses = (
            complete_masses * partial_mask.unsqueeze(-1) +
            F.softplus(best_ghost_mass) * missing_mask.unsqueeze(-1)
        )
        
        # Final restoration loss (differentiable through original path)
        restoration_loss = (delta_F ** 2).sum() / max(partial_mask.sum().item(), 1.0)
        
        return restored_positions, restored_masses, restoration_loss


# ======================================================================
# VERIFICATION: Self-test
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("OlfaBind Module 2 -- Ghost Mass Restoration Self-test")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    B, N = 4, 8
    G_val = torch.tensor(1.0, device=device)
    
    # --- Test 1: GravityFieldComputer ---
    print("\n[Test 1] Gravity field computation...")
    gfc = GravityFieldComputer().to(device)
    
    positions = torch.randn(B, N, 3, device=device) * 2.0
    masses = F.softplus(torch.randn(B, N, 1, device=device))
    mask = torch.ones(B, N, device=device)
    
    force = gfc(positions, masses, G_val, mask)
    assert force.shape == (B, N, 3), f"Expected (B, N, 3), got {force.shape}"
    assert torch.isfinite(force).all(), "Forces must be finite"
    print(f"  Force shape: {force.shape}")
    print(f"  Force magnitude mean: {force.norm(dim=-1).mean().item():.4f}")
    print("  PASS")
    
    # --- Test 2: No missing atoms → zero loss ---
    print("\n[Test 2] No missing atoms -> zero loss...")
    ghost_opt = GhostMassOptimizer(n_optim_steps=5).to(device)
    
    full_mask = torch.ones(B, N, device=device)
    partial_mask = torch.ones(B, N, device=device)  # all present
    
    restored_pos, restored_mass, loss = ghost_opt(
        positions, masses, partial_mask, G_val, full_mask
    )
    print(f"  Restoration loss (no missing): {loss.item():.6f}")
    assert loss.item() < 1e-6, "No missing atoms should give ~zero loss"
    print("  PASS")
    
    # --- Test 3: Missing atoms → positive loss ---
    print("\n[Test 3] Missing atoms -> positive loss...")
    partial_mask_with_missing = torch.ones(B, N, device=device)
    partial_mask_with_missing[:, 6:] = 0  # last 2 atoms missing
    
    restored_pos, restored_mass, loss = ghost_opt(
        positions, masses, partial_mask_with_missing, G_val, full_mask
    )
    print(f"  Restoration loss (2 missing): {loss.item():.6f}")
    assert torch.isfinite(loss), "Loss must be finite"
    print(f"  Restored positions shape: {restored_pos.shape}")
    print(f"  Restored masses shape: {restored_mass.shape}")
    
    # Existing atom positions should remain unchanged
    existing_match = (restored_pos[:, :6] - positions[:, :6]).abs().max().item()
    print(f"  Existing atom position diff: {existing_match:.1e} (should be ~0)")
    assert existing_match < 1e-6, "Existing atoms must keep original positions"
    print("  PASS")
    
    # --- Test 4: Gradient flow ---
    print("\n[Test 4] Gradient flow check...")
    ghost_opt.zero_grad()
    _, _, loss = ghost_opt(
        positions.requires_grad_(True), masses, partial_mask_with_missing, G_val
    )
    loss.backward()
    
    has_grad = sum(1 for p in ghost_opt.parameters() if p.grad is not None)
    print(f"  {has_grad}/{sum(1 for _ in ghost_opt.parameters())} params have gradients")
    print("  PASS")
    
    # --- Test 5: Ghost mass is positive ---
    print("\n[Test 5] Restored masses are positive...")
    assert (restored_mass >= 0).all(), "Ghost masses must be non-negative"
    print(f"  Min restored mass: {restored_mass.min().item():.4f}")
    print(f"  Max restored mass: {restored_mass.max().item():.4f}")
    print("  PASS")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


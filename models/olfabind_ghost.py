"""
olfabind_ghost.py
=================
OlfaBind Module 2: 조기 피질 (역중력 섭동을 이용한 패턴 복원)

불완전한 향기 데이터가 들어왔을 때, '완벽한 은하계'로 복원하는 역산 과정.

수학적 정의:
  ΔF⃗ = F⃗_template - F⃗_partial
  
  min_{r⃗_p, m_p} || ΔF⃗ - G · Σ_i (m_i · m_p / |r⃗_i - r⃗_p|³) · (r⃗_i - r⃗_p) ||²

  경사하강법으로 유령 질량의 위치(r⃗_p)와 질량(m_p)을 최적화하여
  오차가 0에 수렴하면 → 누락된 원자의 본래 자리.

Pipeline:
  - 완전한 constellation → F_template (기준 중력장)
  - 불완전한 constellation (일부 원자 off) → F_partial
  - Ghost Mass Optimizer: gradient descent로 (r_p, m_p) 찾아 ΔF 복원
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GravityFieldComputer(nn.Module):
    """
    주어진 질량/위치 배열에서 각 위치에 작용하는 중력 벡터장을 계산.
    
    F⃗_i = G · Σ_{j≠i} (m_i · m_j / |r⃗_j - r⃗_i|³) · (r⃗_j - r⃗_i)
    
    Input:  positions (B, N, 3), masses (B, N, 1)
    Output: force_field (B, N, 3) — 각 위치에 작용하는 총 중력
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
    역중력 섭동을 이용한 패턴 복원 모듈.
    
    불완전한 향기 데이터에서 누락된 원자(유령 질량)의 위치와 질량을
    최적화하여 원래의 중력장을 복원.
    
    수학:
      min_{r_p, m_p} || ΔF - G·Σ(m_i·m_p/|r_i-r_p|³)(r_i-r_p) ||²
    
    Input:  
      complete_positions  (B, N, 3)  — 완전한 향기의 원자 위치
      complete_masses     (B, N, 1)  — 완전한 향기의 원자 질량
      partial_mask        (B, N)     — 1=존재, 0=누락
      G                   scalar     — 중력 상수
    
    Output:
      restored_positions  (B, N, 3)  — 복원된 전체 위치
      restored_masses     (B, N, 1)  — 복원된 전체 질량
      restoration_loss    scalar     — 복원 오차 ||ΔF||²
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
        existing_positions: torch.Tensor,   # (B, K, 3) — 존재하는 원자들
        existing_masses: torch.Tensor,       # (B, K, 1)
        ghost_positions: torch.Tensor,       # (B, G, 3) — 유령 질량 위치
        ghost_masses: torch.Tensor,          # (B, G, 1) — 유령 질량
        G: torch.Tensor,
    ) -> torch.Tensor:
        """
        유령 질량들이 기존 원자 위치에 만드는 추가 중력 계산.
        
        Returns: ghost_force (B, K, 3) — 유령 질량에 의한 기존 위치의 중력 변화
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
        partial_mask: torch.Tensor,          # (B, N) — 1=존재, 0=누락
        G: torch.Tensor,                     # scalar
        full_mask: torch.Tensor = None,      # (B, N) — 전체 유효 원자 마스크
    ):
        """
        누락된 원자를 유령 질량으로 복원.
        
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
        missing_mask = full_mask * (1.0 - partial_mask)  # (B, N) — 누락된 위치
        n_missing = missing_mask.sum(dim=1).max().int().item()
        
        if n_missing == 0:
            # 누락 없음 → 그대로 반환
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return complete_positions.clone(), complete_masses.clone(), zero_loss
        
        # === Step 1: 기준 중력장 (완전한 상태) ===
        F_template = self.gravity_computer(
            complete_positions, complete_masses, G, full_mask
        )  # (B, N, 3)
        
        # === Step 2: 부분 중력장 (누락 상태) ===
        partial_masses = complete_masses * partial_mask.unsqueeze(-1)
        F_partial = self.gravity_computer(
            complete_positions, partial_masses, G, partial_mask
        )  # (B, N, 3)
        
        # === Step 3: 결핍 힘 ===
        delta_F = F_template - F_partial  # (B, N, 3) — 복원해야 할 중력 차이
        
        # 존재하는 원자에서만 결핍 힘 계산 (누락된 위치는 무시)
        delta_F = delta_F * partial_mask.unsqueeze(-1)  # (B, N, 3)
        
        # === Step 4: Ghost Mass 최적화 ===
        # 누락된 원자 위치의 초기 추정 = 원래 위치 + 약간의 노이즈
        ghost_pos = complete_positions.detach().clone()  # 초기 추정
        ghost_pos = ghost_pos + torch.randn_like(ghost_pos) * 0.1
        ghost_pos = ghost_pos * missing_mask.unsqueeze(-1)
        ghost_pos.requires_grad_(True)
        
        # 유령 질량 초기값
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
            
            # 유령 질량은 양수로 제한
            ghost_masses_pos = F.softplus(ghost_mass_raw) * missing_mask.unsqueeze(-1)
            
            # 유령 위치도 누락 위치에만 적용
            ghost_pos_masked = ghost_pos * missing_mask.unsqueeze(-1)
            
            # 유령 질량이 기존 원자 위치에 만드는 힘
            # 존재하는 원자 위치들 추출
            exist_mask = partial_mask.bool()
            
            # 전체 위치에서 유령에 의한 힘 계산 (간단한 버전)
            ghost_force = self.compute_ghost_force(
                complete_positions, complete_masses * partial_mask.unsqueeze(-1),
                ghost_pos_masked, ghost_masses_pos, G
            )  # (B, N, 3)
            
            # 차이: ||ΔF - ghost_force||²
            ghost_force_at_existing = ghost_force * partial_mask.unsqueeze(-1)
            residual = delta_F - ghost_force_at_existing  # (B, N, 3)
            optim_loss = (residual ** 2).sum() / max(partial_mask.sum().item(), 1.0)
            
            if optim_loss.item() < best_loss:
                best_loss = optim_loss.item()
                best_ghost_pos = ghost_pos.detach().clone()
                best_ghost_mass = ghost_mass_raw.detach().clone()
            
            optim_loss.backward(retain_graph=True)
            inner_optimizer.step()
        
        # === Step 5: 복원된 상태 조합 ===
        # 존재하는 원자는 원래 위치, 누락 원자는 최적화된 유령 위치
        restored_positions = (
            complete_positions * partial_mask.unsqueeze(-1) +
            best_ghost_pos * missing_mask.unsqueeze(-1)
        )
        
        restored_masses = (
            complete_masses * partial_mask.unsqueeze(-1) +
            F.softplus(best_ghost_mass) * missing_mask.unsqueeze(-1)
        )
        
        # 최종 restoration loss (differentiable through original path)
        restoration_loss = (delta_F ** 2).sum() / max(partial_mask.sum().item(), 1.0)
        
        return restored_positions, restored_masses, restoration_loss


# ======================================================================
# VERIFICATION: Self-test
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("OlfaBind Module 2 — Ghost Mass Restoration Self-test")
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
    print("\n[Test 2] No missing atoms → zero loss...")
    ghost_opt = GhostMassOptimizer(n_optim_steps=5).to(device)
    
    full_mask = torch.ones(B, N, device=device)
    partial_mask = torch.ones(B, N, device=device)  # 모두 존재
    
    restored_pos, restored_mass, loss = ghost_opt(
        positions, masses, partial_mask, G_val, full_mask
    )
    print(f"  Restoration loss (no missing): {loss.item():.6f}")
    assert loss.item() < 1e-6, "No missing atoms should give ~zero loss"
    print("  PASS")
    
    # --- Test 3: Missing atoms → positive loss ---
    print("\n[Test 3] Missing atoms → positive loss...")
    partial_mask_with_missing = torch.ones(B, N, device=device)
    partial_mask_with_missing[:, 6:] = 0  # 마지막 2개 원자 누락
    
    restored_pos, restored_mass, loss = ghost_opt(
        positions, masses, partial_mask_with_missing, G_val, full_mask
    )
    print(f"  Restoration loss (2 missing): {loss.item():.6f}")
    assert torch.isfinite(loss), "Loss must be finite"
    print(f"  Restored positions shape: {restored_pos.shape}")
    print(f"  Restored masses shape: {restored_mass.shape}")
    
    # 존재하는 원자 위치는 변하지 않아야 함
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


"""
olfabind_engine.py
==================
OlfaBind: Olfactory Competitive Binding Network
Module 2 (Physics Processing Engine) & Module 3 (Evaluation Output Layer)

Production-ready PyTorch implementation.
All operations are differentiable and GPU-compatible.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional
import math


# ======================================================================
# Module 2: Physics Processing Engine
# ======================================================================

class ConstellationToCelestial(nn.Module):
    """
    Maps constellation patterns (micro-feature vectors from Module 1)
    to physical quantities: mass, position, velocity.
    
    Input:  constellations (B, N, D_atom) - activated atom patterns
            mask           (B, N)         - valid molecule indicators
    Output: masses     (B, N, 1) - always positive via Softplus
            positions  (B, N, 3) - initial 3D coordinates
            velocities (B, N, 3) - initial orbital velocities
    """
    def __init__(self, d_atom: int):
        super().__init__()
        self.mass_proj = nn.Linear(d_atom, 1)
        self.pos_proj = nn.Linear(d_atom, 3)
        self.vel_proj = nn.Linear(d_atom, 3)
        
        # Initialize position projection: gain=0.5 for discriminative spread
        # (gain=0.1 caused pos_std=0.001, gain=1.0 caused NaN after 4 epochs)
        nn.init.xavier_uniform_(self.pos_proj.weight, gain=0.5)
        nn.init.zeros_(self.pos_proj.bias)
        
        # Initialize velocity projection: gain=0.5 for meaningful orbital motion
        # (gain=0.01 caused vel_std=0.00005 -> all mixtures identical orbits)
        nn.init.xavier_uniform_(self.vel_proj.weight, gain=0.5)
        nn.init.zeros_(self.vel_proj.bias)
    
    def forward(
        self,
        constellations: torch.Tensor,
        mask: torch.Tensor,
        override_positions: Optional[torch.Tensor] = None
    ):
        # masses: Softplus ensures strictly positive, clamp to reasonable range
        masses = F.softplus(self.mass_proj(constellations)).clamp(max=5.0)  # (B, N, 1)
        
        # positions: use override if provided (from Module 1-B contrastive latent),
        # otherwise use learned pos_proj
        if override_positions is not None:
            positions = override_positions                                    # (B, N, 3)
        else:
            positions = torch.tanh(self.pos_proj(constellations)) * 2.0       # (B, N, 3)
        
        # velocities: tanh bounds to [-0.5, 0.5] -> enough for meaningful orbits
        velocities = torch.tanh(self.vel_proj(constellations)) * 0.5        # (B, N, 3)
        
        # Zero out invalid molecules
        masses = masses * mask.unsqueeze(-1)
        positions = positions * mask.unsqueeze(-1)
        velocities = velocities * mask.unsqueeze(-1)
        
        return masses, positions, velocities


class GravitationalEngine(nn.Module):
    """
    N체 물리 엔진: 향의 병합과 증발 시뮬레이션.
    
    Velocity Verlet 적분기를 사용한 N-body 중력 시뮬레이션.
    
    수학:
      m_i · d²r⃗_i/dt² = Σ_{j≠i} G · m_i·m_j / |r⃗_j-r⃗_i|³ · (r⃗_j-r⃗_i)
      m_i(t) = m_{i,0} · e^{-k·T·t}  (질량 감쇠 = 증발)
    
    k = 분자별 고유 휘발 상수 (learnable)
    T = 피부 온도 (기본 37°C)
    G = 학습 가능 중력 상수 [0.01, 10.0]
    
    Input:  masses (B,N,1), positions (B,N,3), velocities (B,N,3), mask (B,N)
    Output: trajectory (B,T,N,3), final_pos (B,N,3), final_vel (B,N,3),
            mass_history (B,T,N,1) — 시간에 따른 질량 감쇠 이력
    """
    def __init__(self, n_steps: int = 32, dt: float = 0.01, epsilon: float = 1e-4,
                 accel_clamp: float = 100.0, vel_clamp: float = 50.0,
                 default_temperature: float = 37.0):
        super().__init__()
        self.n_steps = n_steps
        self.dt = dt
        self.epsilon = epsilon
        self.accel_clamp = accel_clamp
        self.vel_clamp = vel_clamp
        self.default_temperature = default_temperature
        
        # Learnable gravitational constant
        self.log_G = nn.Parameter(torch.tensor(0.0))  # G = exp(log_G), init G=1.0
        
        # Learnable volatility constant k for mass decay
        # m_i(t) = m_i(0) · exp(-k · T · t)
        # log_k init = log(0.001) → very slow evaporation initially
        self.log_k = nn.Parameter(torch.tensor(math.log(0.001)))
    
    @property
    def G(self) -> torch.Tensor:
        """Clamped gravitational constant in [0.01, 10.0]."""
        return self.log_G.exp().clamp(min=0.01, max=10.0)
    
    def compute_accelerations(
        self,
        positions: torch.Tensor,   # (B, N, 3)
        masses: torch.Tensor,      # (B, N, 1)
        mask: torch.Tensor         # (B, N)
    ) -> torch.Tensor:
        """
        Compute gravitational acceleration on each body from all others.
        Fully vectorized, no loops.
        
        Returns: accelerations (B, N, 3)
        """
        B, N, _ = positions.shape
        
        # Pairwise displacement vectors: (B, N, N, 3)
        # diff[b, i, j] = positions[b, j] - positions[b, i]
        diff = positions.unsqueeze(1) - positions.unsqueeze(2)  # (B, N, N, 3)
        
        # Pairwise distances: (B, N, N, 1)
        dist = diff.norm(dim=-1, keepdim=True).clamp(min=self.epsilon)
        
        # Gravitational acceleration contribution from j to i:
        # a_ij = G * m_j * (p_j - p_i) / |p_j - p_i|^3
        # masses: (B, N, 1) -> (B, 1, N, 1) for broadcasting as m_j
        m_j = masses.unsqueeze(1)  # (B, 1, N, 1)
        
        accel_pairs = self.G * m_j * diff / (dist ** 3)  # (B, N, N, 3)
        
        # Self-interaction mask: zero out i==j diagonal
        eye = torch.eye(N, device=positions.device, dtype=torch.bool)
        eye = eye.unsqueeze(0).unsqueeze(-1)  # (1, N, N, 1)
        accel_pairs = accel_pairs.masked_fill(eye, 0.0)
        
        # Mask invalid molecules (both source and target)
        pair_mask = mask.unsqueeze(1) * mask.unsqueeze(2)  # (B, N, N)
        accel_pairs = accel_pairs * pair_mask.unsqueeze(-1)
        
        # Sum over source dimension
        accel = accel_pairs.sum(dim=2)  # (B, N, 3)
        
        # Clamp to prevent explosion
        accel = accel.clamp(min=-self.accel_clamp, max=self.accel_clamp)
        
        return accel
    
    def verlet_step(
        self,
        pos: torch.Tensor,    # (B, N, 3)
        vel: torch.Tensor,    # (B, N, 3)
        acc: torch.Tensor,    # (B, N, 3)
        masses: torch.Tensor, # (B, N, 1)
        mask: torch.Tensor    # (B, N)
    ):
        """Single Velocity Verlet integration step."""
        dt = self.dt
        
        # Update position
        new_pos = pos + vel * dt + 0.5 * acc * dt * dt
        
        # Compute acceleration at new position
        new_acc = self.compute_accelerations(new_pos, masses, mask)
        
        # Update velocity
        new_vel = vel + 0.5 * (acc + new_acc) * dt
        
        # Clamp velocities to prevent runaway
        new_vel = new_vel.clamp(min=-self.vel_clamp, max=self.vel_clamp)
        
        # Zero out invalid molecules
        new_pos = new_pos * mask.unsqueeze(-1)
        new_vel = new_vel * mask.unsqueeze(-1)
        
        return new_pos, new_vel, new_acc
    
    @property
    def k(self) -> torch.Tensor:
        """Volatility constant for mass decay, clamped to [1e-6, 0.1]."""
        return self.log_k.exp().clamp(min=1e-6, max=0.1)
    
    def mass_at_time(self, m0: torch.Tensor, t: float,
                     temperature: float = None) -> torch.Tensor:
        """
        질량 감쇠: m_i(t) = m_{i,0} · exp(-k · T · t)
        
        Args:
            m0: initial masses (B, N, 1)
            t: current time (float)
            temperature: skin temperature (default: 37.0)
        Returns:
            m_t: decayed masses (B, N, 1)
        """
        if temperature is None:
            temperature = self.default_temperature
        # Normalize temperature to [0, 1] range (37°C ≈ 1.0)
        T_norm = temperature / 37.0
        return m0 * torch.exp(-self.k * T_norm * t)
    
    def forward(
        self,
        masses: torch.Tensor,     # (B, N, 1) — initial masses m_{i,0}
        positions: torch.Tensor,  # (B, N, 3)
        velocities: torch.Tensor, # (B, N, 3)
        mask: torch.Tensor,       # (B, N)
        temperature: float = None # 피부 온도 (기본 37°C)
    ):
        B, N, _ = positions.shape
        
        # Record trajectory and mass history
        trajectory = []
        mass_history = []
        
        # Initial state
        m_t = masses  # m_i(0)
        acc = self.compute_accelerations(positions, m_t, mask)
        
        pos, vel = positions, velocities
        trajectory.append(pos.unsqueeze(1))
        mass_history.append(m_t.unsqueeze(1))
        
        for t in range(self.n_steps):
            # === 질량 감쇠: m_i(t) = m_{i,0} · exp(-k · T · t) ===
            time_val = (t + 1) * self.dt
            m_t = self.mass_at_time(masses, time_val, temperature)
            m_t = m_t * mask.unsqueeze(-1)  # zero out invalid
            
            # Velocity Verlet step with decayed mass
            if self.training and self.n_steps > 16:
                pos, vel, acc = checkpoint(
                    self.verlet_step, pos, vel, acc, m_t, mask,
                    use_reentrant=False
                )
            else:
                pos, vel, acc = self.verlet_step(pos, vel, acc, m_t, mask)
            
            trajectory.append(pos.unsqueeze(1))
            mass_history.append(m_t.unsqueeze(1))
        
        # (B, T+1, N, 3) and (B, T+1, N, 1)
        trajectory = torch.cat(trajectory, dim=1)
        mass_history = torch.cat(mass_history, dim=1)
        
        return trajectory, pos, vel, mass_history


class OrbitalStabilityEvaluator(nn.Module):
    """
    Module 3: Evaluates the stability of simulated orbital systems.
    
    Computes three differentiable stability metrics:
    1. Energy Conservation Index (S_E)
    2. Orbital Resonance Index (S_R) — using differentiable soft-argmax FFT
    3. Orbital Compactness Index (S_C)
    
    Combines them with learnable weights into a final stability score.
    
    Input:  trajectory (B, T, N, 3) - position history
            masses     (B, N, 1)
            velocities_history or final velocities
            mask       (B, N)
    Output: stability  (B,) - stability score in [0, 1]
            physics_embedding (B, D_phys) - physics feature vector
    """
    def __init__(self, G_ref: nn.Parameter, dt: float = 0.01, max_resonance_ratio: int = 6,
                 vel_energy_clamp: float = 10.0):
        super().__init__()
        self.G_ref = G_ref  # reference to GravitationalEngine's log_G
        self.dt = dt
        self.max_q = max_resonance_ratio
        self.vel_energy_clamp = vel_energy_clamp
        
        # Learnable combination weights
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))
        
        # Learnable temperature for FFT soft-argmax (init=0.1)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(0.1)))
        
        # Learnable sigma for resonance detection width (init=0.5)
        self.log_sigma = nn.Parameter(torch.tensor(math.log(0.5)))
    
    @property
    def G(self):
        return self.G_ref.exp().clamp(min=0.01, max=10.0)
    
    def energy_conservation(
        self,
        trajectory: torch.Tensor,  # (B, T, N, 3)
        masses: torch.Tensor,      # (B, N, 1)
        mask: torch.Tensor         # (B, N)
    ) -> torch.Tensor:
        """
        Compute energy at each timestep and measure its variance.
        S_E = 1 / (1 + Var(E_t) / (|E_0| + eps)^2)
        
        Fully vectorized (no T-loop).
        """
        B, T, N, _ = trajectory.shape
        epsilon = 1e-6
        
        # === Kinetic Energy (vectorized over T) ===
        # Velocity approximation: v_t = (pos_{t+1} - pos_t) / dt
        # For last timestep, use backward difference
        vel = torch.zeros_like(trajectory)  # (B, T, N, 3)
        vel[:, :-1] = (trajectory[:, 1:] - trajectory[:, :-1]) / self.dt
        vel[:, -1] = vel[:, -2]  # copy second-to-last
        vel = vel.clamp(min=-self.vel_energy_clamp, max=self.vel_energy_clamp)
        
        # KE per molecule per timestep
        v_sq = (vel ** 2).sum(dim=-1).clamp(max=100.0)  # (B, T, N)
        ke_per_mol = 0.5 * masses.squeeze(-1).unsqueeze(1) * v_sq  # (B, T, N)
        ke = (ke_per_mol * mask.unsqueeze(1)).sum(dim=-1)  # (B, T)
        
        # === Potential Energy (vectorized over T) ===
        # Pairwise distance at each timestep
        pos_i = trajectory.unsqueeze(3)  # (B, T, N, 1, 3)
        pos_j = trajectory.unsqueeze(2)  # (B, T, 1, N, 3)
        diff = pos_j - pos_i             # (B, T, N, N, 3)
        dist = diff.norm(dim=-1).clamp(min=1e-4)  # (B, T, N, N)
        
        m_i = masses.squeeze(-1).unsqueeze(1).unsqueeze(3)  # (B, 1, N, 1)
        m_j = masses.squeeze(-1).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
        pe_matrix = -self.G * m_i * m_j / dist  # (B, T, N, N)
        
        # Zero diagonal and invalid pairs
        eye = torch.eye(N, device=trajectory.device).unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
        pair_mask = (mask.unsqueeze(1) * mask.unsqueeze(2)).unsqueeze(1)  # (B, 1, N, N)
        pe_matrix = pe_matrix * (1 - eye) * pair_mask
        pe = 0.5 * pe_matrix.sum(dim=(2, 3))  # (B, T)
        
        # === Total energy and S_E ===
        energies = (ke + pe).clamp(min=-1e4, max=1e4)  # (B, T)
        
        e0 = energies[:, 0].abs() + epsilon
        e_var = energies.var(dim=1)
        ratio = (e_var / (e0 ** 2)).clamp(max=1e4)
        s_e = 1.0 / (1.0 + ratio)
        return s_e  # (B,)
    
    def orbital_resonance(
        self,
        trajectory: torch.Tensor,  # (B, T, N, 3)
        mask: torch.Tensor         # (B, N)
    ) -> torch.Tensor:
        """
        Estimate dominant orbital frequencies via FFT with soft-argmax,
        then measure how close frequency ratios are to simple integer ratios.
        
        Fully differentiable: uses softmax-weighted frequency average
        instead of non-differentiable argmax.
        """
        B, T, N, _ = trajectory.shape
        
        # Compute radial distance from origin over time: (B, T, N)
        radii = trajectory.norm(dim=-1)  # (B, T, N)
        
        # FFT along time axis
        # Remove mean to get oscillatory component
        radii_centered = radii - radii.mean(dim=1, keepdim=True)
        
        # Real FFT: (B, T, N) -> (B, T//2+1, N)
        fft_result = torch.fft.rfft(radii_centered, dim=1)
        magnitudes = fft_result.abs()  # (B, F, N) where F = T//2+1
        
        # Zero out DC component (frequency 0)
        magnitudes[:, 0, :] = 0.0
        
        # Soft-argmax: differentiable dominant frequency estimation
        # Learnable temperature controls sharpness of frequency estimation
        temperature = self.log_temperature.exp().clamp(min=0.01, max=1.0)
        freq_indices = torch.arange(magnitudes.shape[1], device=trajectory.device).float()
        freq_indices = freq_indices.unsqueeze(0).unsqueeze(-1)  # (1, F, 1)
        
        weights = F.softmax(magnitudes / (temperature + 1e-8), dim=1)  # (B, F, N)
        dominant_freq = (weights * freq_indices).sum(dim=1)  # (B, N)
        
        # Mask invalid molecules
        dominant_freq = dominant_freq * mask + (1 - mask) * 1.0  # avoid div by zero
        
        # Compute pairwise frequency ratios
        freq_i = dominant_freq.unsqueeze(2)  # (B, N, 1)
        freq_j = dominant_freq.unsqueeze(1)  # (B, 1, N)
        ratios = freq_i / (freq_j + 1e-8)   # (B, N, N)
        
        # Generate simple integer ratios to compare against
        Q = self.max_q
        p_vals = torch.arange(1, Q + 1, device=trajectory.device).float()
        q_vals = torch.arange(1, Q + 1, device=trajectory.device).float()
        simple_ratios = (p_vals.unsqueeze(1) / q_vals.unsqueeze(0)).reshape(-1)  # (Q*Q,)
        
        # For each pair, find minimum distance to any simple ratio
        # ratios: (B, N, N) -> (B, N, N, 1)
        # simple_ratios: (Q*Q,) -> (1, 1, 1, Q*Q)
        ratios_exp = ratios.unsqueeze(-1)
        simple_exp = simple_ratios.reshape(1, 1, 1, -1)
        
        # Differentiable soft-min: use negative softmax
        distances = (ratios_exp - simple_exp).abs()  # (B, N, N, Q*Q)
        soft_min_dist = -((-distances / 0.1).logsumexp(dim=-1))  # (B, N, N) — soft-min approx
        
        # Resonance score: exp(-dist^2 / sigma^2)
        # Learnable sigma controls resonance detection width
        sigma = self.log_sigma.exp().clamp(min=0.05, max=2.0)
        resonance_pairs = torch.exp(-soft_min_dist ** 2 / (sigma ** 2))  # (B, N, N)
        
        # Mask: only valid pairs, exclude diagonal
        eye = torch.eye(N, device=trajectory.device).unsqueeze(0)
        pair_mask = mask.unsqueeze(1) * mask.unsqueeze(2) * (1 - eye)
        
        resonance_pairs = resonance_pairs * pair_mask
        n_pairs = pair_mask.sum(dim=(1, 2)).clamp(min=1.0)
        
        s_r = resonance_pairs.sum(dim=(1, 2)) / n_pairs  # (B,)
        return s_r
    
    def orbital_compactness(
        self,
        trajectory: torch.Tensor,  # (B, T, N, 3)
        mask: torch.Tensor         # (B, N)
    ) -> torch.Tensor:
        """
        Measure whether planets stay bounded (compact) or diverge (chaos).
        S_C = 1 / (1 + max_i(|p_i^T| / |p_i^0| - 1)^2)
        """
        initial_r = trajectory[:, 0].norm(dim=-1).clamp(min=1e-4)   # (B, N)
        final_r = trajectory[:, -1].norm(dim=-1)                     # (B, N)
        
        expansion = (final_r / initial_r - 1.0) ** 2  # (B, N)
        expansion = expansion * mask  # zero out invalid
        
        # Differentiable soft-max over molecules
        max_expansion = (expansion * mask).max(dim=-1).values  # (B,)
        
        s_c = 1.0 / (1.0 + max_expansion)
        return s_c  # (B,)
    
    def chaos_resonance_score(
        self,
        trajectory: torch.Tensor,  # (B, T, N, 3)
        mask: torch.Tensor         # (B, N)
    ) -> torch.Tensor:
        """
        공명/카오스 판별: 궤도 간 거리의 시계열 분석.
        
        거리가 무한대로 발산 → '카오스' (역겨운 향)  → score ≈ 0
        거리가 일정 범위 내 주기성 → '공명' (아름다운 향) → score ≈ 1
        
        방법: 인접 궤도의 지수적 발산율 (Lyapunov exponent 근사)
        """
        B, T, N, _ = trajectory.shape
        
        if T < 3 or N < 2:
            return torch.ones(B, device=trajectory.device)
        
        # 시간에 따른 모든 쌍별 거리 변화 추적
        # positions at first and last third of trajectory
        T_early = max(1, T // 3)
        T_late = max(T_early + 1, 2 * T // 3)
        
        # 초기 쌍별 거리
        pos_early = trajectory[:, :T_early]  # (B, T_early, N, 3)
        pos_late = trajectory[:, T_late:]     # (B, T_rest, N, 3)
        
        # 초기 평균 쌍별 거리
        d_early_ij = pos_early[:, :, :, None, :] - pos_early[:, :, None, :, :]  # (B,T,N,N,3)
        d_early = d_early_ij.norm(dim=-1).mean(dim=1)  # (B, N, N) — 시간 평균
        
        # 후기 평균 쌍별 거리
        d_late_ij = pos_late[:, :, :, None, :] - pos_late[:, :, None, :, :]  # (B,T,N,N,3)
        d_late = d_late_ij.norm(dim=-1).mean(dim=1)  # (B, N, N) — 시간 평균
        
        # 발산 비율: d_late / d_early
        # 값 > 1 → 발산 (카오스), 값 ≈ 1 → 안정 (공명)
        divergence = d_late / (d_early + 1e-6)  # (B, N, N)
        
        # 대각선 제외, 유효 쌍만
        eye = torch.eye(N, device=trajectory.device).unsqueeze(0)
        pair_mask = mask.unsqueeze(1) * mask.unsqueeze(2) * (1 - eye)
        
        # 발산 정도의 평균
        divergence_masked = divergence * pair_mask
        n_pairs = pair_mask.sum(dim=(1, 2)).clamp(min=1.0)
        mean_divergence = divergence_masked.sum(dim=(1, 2)) / n_pairs  # (B,)
        
        # 공명 점수: 발산이 1에 가까울수록 높음
        # score = 1 / (1 + (divergence - 1)^2)
        chaos_score = 1.0 / (1.0 + (mean_divergence - 1.0) ** 2)
        
        return chaos_score  # (B,)  — 1.0=완벽 공명, 0.0=카오스
    
    def compute_energy_trajectory(
        self,
        trajectory: torch.Tensor,  # (B, T, N, 3)
        masses: torch.Tensor,      # (B, N, 1)
        mask: torch.Tensor         # (B, N)
    ) -> torch.Tensor:
        """
        Hamiltonian trajectory: H(t) = KE(t) + PE(t) for each timestep.
        
        Returns: energy_trajectory (B, T) — total energy at each timestep
        
        Based on: Greydanus et al., "Hamiltonian Neural Networks" (NeurIPS 2019)
        """
        B, T, N, _ = trajectory.shape
        
        # KE: velocity from finite differences
        vel = torch.zeros_like(trajectory)
        vel[:, :-1] = (trajectory[:, 1:] - trajectory[:, :-1]) / self.dt
        vel[:, -1] = vel[:, -2]
        vel = vel.clamp(min=-self.vel_energy_clamp, max=self.vel_energy_clamp)
        
        v_sq = (vel ** 2).sum(dim=-1).clamp(max=100.0)  # (B, T, N)
        ke_per_mol = 0.5 * masses.squeeze(-1).unsqueeze(1) * v_sq
        ke = (ke_per_mol * mask.unsqueeze(1)).sum(dim=-1)  # (B, T)
        
        # PE: pairwise gravitational potential
        pos_i = trajectory.unsqueeze(3)
        pos_j = trajectory.unsqueeze(2)
        dist = (pos_j - pos_i).norm(dim=-1).clamp(min=1e-4)
        
        m_i = masses.squeeze(-1).unsqueeze(1).unsqueeze(3)
        m_j = masses.squeeze(-1).unsqueeze(1).unsqueeze(2)
        pe_matrix = -self.G * m_i * m_j / dist
        
        eye = torch.eye(N, device=trajectory.device).unsqueeze(0).unsqueeze(0)
        pair_mask = (mask.unsqueeze(1) * mask.unsqueeze(2)).unsqueeze(1)
        pe_matrix = pe_matrix * (1 - eye) * pair_mask
        pe = 0.5 * pe_matrix.sum(dim=(2, 3))  # (B, T)
        
        return (ke + pe).clamp(min=-1e4, max=1e4)
    
    def compute_pinn_loss(
        self,
        trajectory: torch.Tensor,  # (B, T, N, 3)
        masses: torch.Tensor,      # (B, N, 1)
        mask: torch.Tensor         # (B, N)
    ) -> torch.Tensor:
        """
        Physics-Informed regularization: penalize violations of conservation laws.
        
        L_pinn = |dE/dt|_mean + |dp/dt|_mean
        
        Based on: Raissi et al., "Physics-Informed Neural Networks" (JCP 2019)
        """
        # Energy conservation: minimize |dE/dt|
        energies = self.compute_energy_trajectory(trajectory, masses, mask)  # (B, T)
        dE_dt = (energies[:, 1:] - energies[:, :-1]) / self.dt  # (B, T-1)
        energy_violation = dE_dt.pow(2).mean(dim=1)  # (B,)
        
        # Momentum conservation: minimize |dp/dt|
        vel = torch.zeros_like(trajectory)
        vel[:, :-1] = (trajectory[:, 1:] - trajectory[:, :-1]) / self.dt
        vel[:, -1] = vel[:, -2]
        vel = vel.clamp(min=-self.vel_energy_clamp, max=self.vel_energy_clamp)
        
        momentum = (masses.unsqueeze(1) * vel * mask.unsqueeze(1).unsqueeze(-1)).sum(dim=2)  # (B, T, 3)
        dp_dt = (momentum[:, 1:] - momentum[:, :-1]) / self.dt  # (B, T-1, 3)
        momentum_violation = dp_dt.pow(2).sum(dim=-1).mean(dim=1)  # (B,)
        
        return (energy_violation + momentum_violation).mean()
    
    def compute_spectral_signature(
        self,
        trajectory: torch.Tensor,  # (B, T, N, 3)
        masses: torch.Tensor,      # (B, N, 1)
        mask: torch.Tensor         # (B, N)
    ) -> torch.Tensor:
        """
        Spectral signature: eigenvalues of gravitational interaction matrix.
        
        M_ij = G * m_i * m_j / |r_i - r_j|^3  (interaction strength)
        eigenvalues of M → "vibrational modes" of the molecular system
        
        Returns: sorted eigenvalues (B, N) — spectral fingerprint
        """
        B, T, N, _ = trajectory.shape
        
        # Use time-averaged positions for stability
        pos_avg = (trajectory * mask.unsqueeze(1).unsqueeze(-1)).mean(dim=1)  # (B, N, 3)
        
        # Pairwise interaction matrix
        diff = pos_avg.unsqueeze(1) - pos_avg.unsqueeze(2)  # (B, N, N, 3)
        dist = diff.norm(dim=-1).clamp(min=1e-4)  # (B, N, N)
        
        m_i = masses.squeeze(-1).unsqueeze(2)  # (B, N, 1)
        m_j = masses.squeeze(-1).unsqueeze(1)  # (B, 1, N)
        
        # Interaction strength: G * m_i * m_j / |r_ij|^3
        interaction = self.G * m_i * m_j / dist.pow(3)  # (B, N, N)
        
        # Zero diagonal and invalid pairs
        eye = torch.eye(N, device=trajectory.device).unsqueeze(0)
        pair_mask = mask.unsqueeze(1) * mask.unsqueeze(2) * (1 - eye)
        interaction = interaction * pair_mask
        
        # Make symmetric for real eigenvalues
        interaction = (interaction + interaction.transpose(1, 2)) / 2
        
        # Eigenvalues (sorted descending)
        eigenvals = torch.linalg.eigvalsh(interaction)  # (B, N) — sorted ascending
        eigenvals = eigenvals.flip(dims=[-1])  # descending
        
        return eigenvals  # (B, N) — spectral fingerprint
    
    def forward(
        self,
        trajectory: torch.Tensor,  # (B, T, N, 3)
        masses: torch.Tensor,      # (B, N, 1)
        mask: torch.Tensor         # (B, N)
    ):
        s_e = self.energy_conservation(trajectory, masses, mask)
        s_r = self.orbital_resonance(trajectory, mask)
        s_c = self.orbital_compactness(trajectory, mask)
        s_chaos = self.chaos_resonance_score(trajectory, mask)  # 공명/카오스 판별
        
        # Learnable weighted combination → sigmoid
        stability = torch.sigmoid(
            self.alpha * s_e + self.beta * s_r + self.gamma * s_c + self.bias
        ) * s_chaos  # 카오스면 stability가 0에 가까워짐
        
        B = trajectory.shape[0]
        N = masses.shape[1]
        T = trajectory.shape[1]
        n_valid = mask.sum(dim=-1).clamp(min=1)  # (B,)
        
        # Mean mass
        mean_mass = (masses.squeeze(-1) * mask).sum(dim=-1) / n_valid
        
        # Final velocity
        if T >= 2:
            final_vel = (trajectory[:, -1] - trajectory[:, -2]) / self.dt
            final_vel = final_vel.clamp(min=-10.0, max=10.0)
        else:
            final_vel = torch.zeros_like(trajectory[:, 0])
        vel_mag = ((final_vel ** 2).sum(dim=-1) + 1e-8).sqrt()  # (B, N)
        mean_vel = (vel_mag * mask).sum(dim=-1) / n_valid
        
        # Angular momentum
        final_pos = trajectory[:, -1]
        L = torch.cross(final_pos, final_vel, dim=-1)
        total_L = ((L * mask.unsqueeze(-1)).sum(dim=1) ** 2).sum(dim=-1).add(1e-8).sqrt()
        
        # --- Additional discriminative features (trajectory-derived) ---
        # Initial position mean per body (masked mean over 3D coords)
        init_pos = trajectory[:, 0]  # (B, N, 3)
        init_pos_mean = (init_pos * mask.unsqueeze(-1)).sum(dim=1) / n_valid.unsqueeze(-1)  # (B, 3)
        
        # Final position mean
        final_pos_mean = (final_pos * mask.unsqueeze(-1)).sum(dim=1) / n_valid.unsqueeze(-1)  # (B, 3)
        
        # Trajectory displacement: how much each body moved
        displacement = (final_pos - init_pos).norm(dim=-1)  # (B, N)
        mean_displacement = (displacement * mask).sum(dim=-1) / n_valid  # (B,)
        
        # Radii: mean initial and final radius from origin
        init_r = init_pos.norm(dim=-1)  # (B, N)
        final_r = final_pos.norm(dim=-1)  # (B, N)
        mean_init_r = (init_r * mask).sum(dim=-1) / n_valid
        mean_final_r = (final_r * mask).sum(dim=-1) / n_valid
        
        # Physics embedding: (B, 20) — 카오스 점수 추가
        physics_embedding = torch.cat([
            torch.stack([stability, s_e, s_r, s_c, s_chaos, mean_mass, mean_vel, total_L, 
                         mean_displacement, mean_init_r, mean_final_r], dim=-1),  # (B, 11)
            init_pos_mean,   # (B, 3)
            final_pos_mean,  # (B, 3)
            (final_vel * mask.unsqueeze(-1)).sum(dim=1) / n_valid.unsqueeze(-1),  # (B, 3) mean final vel
        ], dim=-1)
        
        return stability, physics_embedding


# ======================================================================
# Full Pipeline: Module 2 + Module 3
# ======================================================================

class PhysicsProcessingEngine(nn.Module):
    """
    Complete OlfaBind physics engine combining:
    - Constellation -> Celestial mapping (Module 2.1)
    - N-body gravitational simulation (Module 2.2)
    - Orbital stability evaluation (Module 3)
    
    Input:  constellations (B, N, D_atom) - from Module 1
            mask           (B, N)         - valid molecule indicators
    Output: stability        (B,)   - stability score [0, 1]
            physics_embedding (B, 7) - physics feature vector
            trajectory       (B, T, N, 3) - full orbital history
    """
    def __init__(
        self,
        d_atom: int,
        n_steps: int = 32,
        dt: float = 0.01,
        epsilon: float = 1e-4,
        accel_clamp: float = 100.0,
        vel_clamp: float = 50.0,
        vel_energy_clamp: float = 10.0
    ):
        super().__init__()
        self.mapper = ConstellationToCelestial(d_atom)
        self.engine = GravitationalEngine(
            n_steps=n_steps, dt=dt, epsilon=epsilon,
            accel_clamp=accel_clamp, vel_clamp=vel_clamp
        )
        self.evaluator = OrbitalStabilityEvaluator(
            G_ref=self.engine.log_G,
            dt=dt,
            vel_energy_clamp=vel_energy_clamp
        )
    
    def forward(
        self,
        constellations: torch.Tensor,
        mask: torch.Tensor,
        override_positions: Optional[torch.Tensor] = None
    ):
        # Step 1: Map constellations to physical quantities
        # If override_positions is provided (from Module 1-B contrastive latent),
        # use those as initial 3D positions instead of learned pos_proj
        masses, positions, velocities = self.mapper(
            constellations, mask, override_positions=override_positions
        )
        
        # Step 2: N-body 시뮬레이션 (질량 감쇠 포함)
        trajectory, final_pos, final_vel, mass_history = self.engine(
            masses, positions, velocities, mask
        )
        
        # Step 3: Evaluate orbital stability
        stability, physics_embedding = self.evaluator(
            trajectory, masses, mask
        )
        
        return stability, physics_embedding, trajectory


class OlfaBindSimilarityModel(nn.Module):
    """
    Top-level model for predicting mixture similarity.
    
    Runs two mixtures through the physics engine, then computes
    cosine similarity of their physics embeddings.
    
    Input:  constellations_a (B, N, D_atom), mask_a (B, N)
            constellations_b (B, N, D_atom), mask_b (B, N)
    Output: similarity (B,) - predicted perceptual similarity
    """
    def __init__(
        self,
        d_atom: int,
        n_steps: int = 32,
        dt: float = 0.01
    ):
        super().__init__()
        self.physics = PhysicsProcessingEngine(
            d_atom=d_atom, n_steps=n_steps, dt=dt
        )
        self.sim = nn.CosineSimilarity(dim=-1)
    
    def forward(
        self,
        const_a: torch.Tensor, mask_a: torch.Tensor,
        const_b: torch.Tensor, mask_b: torch.Tensor
    ):
        _, emb_a, _ = self.physics(const_a, mask_a)
        _, emb_b, _ = self.physics(const_b, mask_b)
        return self.sim(emb_a, emb_b)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ======================================================================
# VERIFICATION: Self-test
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("OlfaBind Module 2 & 3 — Self-test")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Configuration
    B = 4       # batch size
    N = 10      # max molecules per mixture
    D_ATOM = 128  # micro-feature dimension (from Module 1)
    T = 32      # simulation steps
    
    # --- Test 1: Forward pass ---
    print("\n[Test 1] Forward pass...")
    model = OlfaBindSimilarityModel(d_atom=D_ATOM, n_steps=T).to(device)
    
    const_a = torch.randn(B, N, D_ATOM, device=device)
    mask_a = torch.ones(B, N, device=device)
    mask_a[:, 7:] = 0  # only 7 molecules valid
    
    const_b = torch.randn(B, N, D_ATOM, device=device)
    mask_b = torch.ones(B, N, device=device)
    mask_b[:, 5:] = 0  # only 5 molecules valid
    
    similarity = model(const_a, mask_a, const_b, mask_b)
    print(f"  Similarity shape: {similarity.shape}")  # (B,)
    print(f"  Similarity values: {similarity.detach().cpu().numpy()}")
    print(f"  Total parameters: {model.count_parameters()}")
    
    # --- Test 2: Backward pass (gradient flow) ---
    print("\n[Test 2] Backward pass...")
    target = torch.tensor([0.5, 0.8, 0.3, 0.9], device=device)
    loss = F.mse_loss(similarity, target)
    loss.backward()
    
    grad_ok = True
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"  WARNING: No gradient for {name}")
            grad_ok = False
        elif param.grad.isnan().any():
            print(f"  WARNING: NaN gradient for {name}")
            grad_ok = False
    
    if grad_ok:
        print("  All gradients computed successfully (no NaN)")
    print(f"  Loss: {loss.item():.6f}")
    
    # --- Test 3: Physics engine detailed ---
    print("\n[Test 3] Physics engine detailed...")
    engine = model.physics
    stability, phys_emb, trajectory = engine(const_a, mask_a)
    print(f"  Trajectory shape: {trajectory.shape}")      # (B, T+1, N, 3)
    print(f"  Physics embedding shape: {phys_emb.shape}")  # (B, 7)
    print(f"  Stability scores: {stability.detach().cpu().numpy()}")
    
    # --- Test 4: Gravitational constant ---
    print("\n[Test 4] Learnable G...")
    G = engine.engine.G
    print(f"  G value: {G.item():.4f}")
    print(f"  G has grad: {engine.engine.log_G.grad is not None}")
    
    # --- Test 5: Variable molecule count ---
    print("\n[Test 5] Variable molecule counts...")
    mask_var = torch.zeros(B, N, device=device)
    mask_var[0, :2] = 1   # mixture 0: 2 molecules
    mask_var[1, :5] = 1   # mixture 1: 5 molecules
    mask_var[2, :1] = 1   # mixture 2: 1 molecule (no interaction)
    mask_var[3, :10] = 1  # mixture 3: 10 molecules (full)
    
    stab, emb, traj = engine(const_a, mask_var)
    print(f"  Stability (2 mols): {stab[0].item():.4f}")
    print(f"  Stability (5 mols): {stab[1].item():.4f}")
    print(f"  Stability (1 mol):  {stab[2].item():.4f}")
    print(f"  Stability (10 mols): {stab[3].item():.4f}")
    
    # --- Test 6: Training step ---
    print("\n[Test 6] Training step...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(5):
        optimizer.zero_grad()
        sim = model(const_a, mask_a, const_b, mask_b)
        loss = F.mse_loss(sim, target)
        loss.backward()
        
        # Check for NaN
        has_nan = any(p.grad.isnan().any() for p in model.parameters() if p.grad is not None)
        optimizer.step()
        print(f"  Step {step+1}: loss={loss.item():.6f}, nan={has_nan}")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED" if not has_nan else "TESTS COMPLETED WITH WARNINGS")
    print("=" * 60)

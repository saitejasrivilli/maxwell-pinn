"""
src/physics/maxwell.py

PDE residuals for the time-harmonic Maxwell equations in TE mode.

Governing equation (curl-curl form):
    ∇×∇×E − k²(r)·E = iωμ₀·J_coil

where k²(r) = ω²μ₀ε₀ − iωμ₀σ(r)  is the complex wave number.

Network output: E = [E_r_re, E_r_im, E_z_re, E_z_im]
(real and imaginary parts of the complex electric field vector)

All derivatives computed via torch.autograd.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
from omegaconf import DictConfig

MU_0    = 1.2566370614e-6   # H/m
EPS_0   = 8.8541878128e-12  # F/m
PI      = 3.141592653589793


def _grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """∂y/∂x via autograd. x must have requires_grad=True."""
    return torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]


class MaxwellTEResidual(nn.Module):
    """
    Computes the PDE residual for the TE-mode curl-curl equation.
    Used as the physics loss term during training.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.omega = 2 * PI * cfg.coil.frequency
        self.pde_weight = cfg.pde_loss_weight if hasattr(cfg, 'pde_loss_weight') else 1.0

    def wave_number_sq(self, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        k²(r) = ω²μ₀ε₀ − iωμ₀σ
        Returns (k2_re, k2_im) separately for real arithmetic.
        """
        k2_re = (self.omega ** 2) * MU_0 * EPS_0
        k2_im = -self.omega * MU_0 * sigma
        return k2_re, k2_im

    def forward(
        self,
        r: torch.Tensor,           # (N,1) requires_grad=True
        z: torch.Tensor,           # (N,1) requires_grad=True
        E_re: torch.Tensor,        # (N,2)  [E_r_re, E_z_re]
        E_im: torch.Tensor,        # (N,2)  [E_r_im, E_z_im]
        sigma: torch.Tensor,       # (N,1)
        J_coil: torch.Tensor,      # (N,1) imaginary part of J_phi (coil current)
    ) -> torch.Tensor:
        """
        Returns scalar PDE residual loss (mean squared).
        """
        E_r_re, E_z_re = E_re[:, 0:1], E_re[:, 1:2]
        E_r_im, E_z_im = E_im[:, 0:1], E_im[:, 1:2]

        k2_re, k2_im = self.wave_number_sq(sigma)

        # ── curl E (azimuthal component B_phi) ──────────────────────────────
        # (∇×E)_phi = ∂E_r/∂z − ∂E_z/∂r
        curlE_re = _grad(E_r_re, z) - _grad(E_z_re, r)
        curlE_im = _grad(E_r_im, z) - _grad(E_z_im, r)

        # ── curl curl E = −∇²E + ∇(∇·E) in cylindrical coords ─────────────
        # For TE mode (azimuthal symmetry, ∂/∂φ = 0):
        # (∇×∇×E)_r = ∂/∂z (∂E_r/∂z − ∂E_z/∂r)  ... simplified
        # (∇×∇×E)_z = −1/r ∂/∂r (r ∂E_z/∂r) + 1/r ∂/∂r (r ∂E_r/∂z) ... simplified
        # Use the identity: curl curl E = −∇²E  (when ∇·E = 0)
        curl2E_r_re = _grad(curlE_re, z)
        curl2E_r_im = _grad(curlE_im, z)
        curl2E_z_re = -(1.0 / (r + 1e-8)) * _grad(r * _grad(curlE_re, r), r)
        curl2E_z_im = -(1.0 / (r + 1e-8)) * _grad(r * _grad(curlE_im, r), r)

        # ── Residual: ∇×∇×E − k²E = iωμ₀J ─────────────────────────────────
        # Split into real/imag parts:
        #   Re: curl²E_re − k2_re·E_re + k2_im·E_im = 0
        #   Im: curl²E_im − k2_re·E_im − k2_im·E_re = ωμ₀J_coil
        src = self.omega * MU_0 * J_coil

        res_r_re = curl2E_r_re - k2_re * E_r_re + k2_im * E_r_im
        res_r_im = curl2E_r_im - k2_re * E_r_im - k2_im * E_r_re - src

        res_z_re = curl2E_z_re - k2_re * E_z_re + k2_im * E_z_im
        res_z_im = curl2E_z_im - k2_re * E_z_im - k2_im * E_z_re

        loss = (res_r_re**2 + res_r_im**2 + res_z_re**2 + res_z_im**2).mean()
        return loss * self.pde_weight

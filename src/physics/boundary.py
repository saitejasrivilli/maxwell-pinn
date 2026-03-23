"""
src/physics/boundary.py

Boundary condition enforcement — hard (ansatz) and soft (penalty).

Hard BC ansatz:
    E_pred(x) = tanh(dist(x, Γ_PEC) / δ) · E_net(x)

    The tanh taper drives E → 0 as x → wall, exactly satisfying
    the PEC condition (tangential E = 0) without requiring any
    boundary collocation points or penalty term.

Soft BC penalty:
    L_bc = λ · mean(||E_tan||²)  evaluated at boundary points.
    λ is annealed from λ_start → λ_end over warmup_epochs.

Both classes expose the same forward() interface so the training
loop is agnostic to which method is selected.
"""

import torch
import torch.nn as nn
from omegaconf import DictConfig
from typing import Tuple


class HardBCAnsatz(nn.Module):
    """
    Wraps a base network and applies a distance-function taper
    to enforce PEC boundary conditions exactly.

    The taper function:
        w(x) = tanh(d(x) / δ)

    where d(x) = signed distance to the nearest PEC surface
    (positive inside domain, zero on wall).

    E_pred = w(x) · E_net(x)

    This guarantees E_pred = 0 on ∂Ω for any weights of E_net,
    so the network never needs to waste capacity fitting the BC.
    """

    def __init__(self, base_net: nn.Module, cfg: DictConfig):
        super().__init__()
        self.net   = base_net
        self.delta = cfg.bc.delta   # sharpness parameter

    def forward(
        self,
        r: torch.Tensor,   # (N,1)
        z: torch.Tensor,   # (N,1)
        params: torch.Tensor,      # (N, P) operating parameter vector
        dist_wall: torch.Tensor,   # (N,1) signed distance to PEC wall
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (E_re, E_im), each (N,2), after applying the BC taper.
        """
        x = torch.cat([r, z, params], dim=-1)
        raw = self.net(x)                       # (N, 4)
        E_re_raw = raw[:, 0:2]
        E_im_raw = raw[:, 2:4]

        # Taper: zero at wall, approaches 1 in the interior
        taper = torch.tanh(dist_wall / self.delta)   # (N,1)

        E_re = taper * E_re_raw
        E_im = taper * E_im_raw
        return E_re, E_im

    def bc_loss(self, *args, **kwargs) -> torch.Tensor:
        """Hard BC has no penalty loss term."""
        return torch.tensor(0.0, requires_grad=False)


class SoftBCPenalty(nn.Module):
    """
    Standard penalty-based BC enforcement.
    Adds λ·||E_tan||² at boundary collocation points to the total loss.
    λ is linearly annealed from λ_start → λ_end over warmup_epochs.
    """

    def __init__(self, base_net: nn.Module, cfg: DictConfig):
        super().__init__()
        self.net            = base_net
        self.lambda_start   = cfg.bc.lambda_start
        self.lambda_end     = cfg.bc.lambda_end
        self.warmup_epochs  = cfg.bc.warmup_epochs
        self.bc_weight      = cfg.bc.bc_loss_weight
        self._epoch         = 0

    def forward(
        self,
        r: torch.Tensor,
        z: torch.Tensor,
        params: torch.Tensor,
        dist_wall: torch.Tensor,   # unused for soft BC but kept for API parity
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([r, z, params], dim=-1)
        raw = self.net(x)
        return raw[:, 0:2], raw[:, 2:4]

    def bc_loss(
        self,
        r_bc: torch.Tensor,        # (M,1) boundary point r-coords
        z_bc: torch.Tensor,        # (M,1)
        params_bc: torch.Tensor,   # (M,P)
        normal: torch.Tensor,      # (M,2) outward normal unit vector [n_r, n_z]
    ) -> torch.Tensor:
        """
        Penalty: λ · mean(||E − (E·n)n||²) at boundary points.
        The tangential component is E − (E·n)n.
        """
        E_re, E_im = self.forward(r_bc, z_bc, params_bc,
                                   dist_wall=torch.zeros_like(r_bc))

        def tan_sq(E: torch.Tensor) -> torch.Tensor:
            En = (E * normal).sum(dim=-1, keepdim=True)  # normal projection
            E_tan = E - En * normal
            return (E_tan ** 2).sum(dim=-1)

        lam = self._current_lambda()
        loss = lam * self.bc_weight * (tan_sq(E_re) + tan_sq(E_im)).mean()
        return loss

    def _current_lambda(self) -> float:
        t = min(self._epoch / max(self.warmup_epochs, 1), 1.0)
        return self.lambda_start + t * (self.lambda_end - self.lambda_start)

    def step_epoch(self):
        """Call once per epoch to advance the λ schedule."""
        self._epoch += 1

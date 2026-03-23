"""
src/geometry/cylindrical.py

Cylindrical chamber geometry: signed-distance functions (SDFs)
and Latin-hypercube collocation point samplers.

All coordinates are (r, z) in metres.  The third input dimension
fed to the network is a flattened operating-parameter vector;
that encoding lives in src/geometry/param_encoder.py.
"""

import torch
import numpy as np
from typing import Dict, Tuple


class CylindricalChamber:
    """
    Cylindrical ICP chamber geometry.

    Domain:   r ∈ [0, R_chamber],  z ∈ [0, H_chamber]
    Plasma:   r ∈ [0, R_plasma]  (subset)
    Wall:     outer boundary — PEC (tangential E = 0)
    """

    def __init__(self, cfg):
        self.R = cfg.chamber.radius
        self.H = cfg.chamber.height
        self.R_plasma = self.R * 0.85  # plasma fills 85% of radius

        self.n_interior  = cfg.domain.n_interior
        self.n_boundary  = cfg.domain.n_boundary
        self.n_interface = cfg.domain.n_interface

    # ── SDFs ────────────────────────────────────────────────────────────────

    def sdf_chamber(self, r: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Signed distance to the chamber wall.
        Negative inside, positive outside.
        """
        d_r = self.R - r                        # distance to radial wall
        d_z_bot = z                             # distance to bottom wall
        d_z_top = self.H - z                    # distance to top wall
        return torch.minimum(torch.minimum(d_r, d_z_bot), d_z_top)

    def sdf_plasma(self, r: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Signed distance to the plasma–vacuum interface."""
        return self.R_plasma - r

    # ── Collocation samplers ─────────────────────────────────────────────────

    def sample_interior(self, n: int = None) -> Dict[str, torch.Tensor]:
        """Latin-hypercube samples inside the chamber (physics residual points)."""
        n = n or self.n_interior
        pts = self._lhs(n, [[0, self.R], [0, self.H]])
        r, z = pts[:, 0:1], pts[:, 1:2]
        # Keep only points inside chamber (rejection sampling for non-rectangular)
        mask = self.sdf_chamber(r, z) > 0
        r, z = r[mask.squeeze()], z[mask.squeeze()]
        return {"r": r, "z": z}

    def sample_boundary(self, n: int = None) -> Dict[str, torch.Tensor]:
        """
        Points on the PEC wall: radial wall + top/bottom endcaps.
        Used for soft BC penalty; not needed for hard BC ansatz.
        """
        n = n or self.n_boundary
        n_each = n // 3

        # Radial wall: r = R, z ~ U[0, H]
        z_rad  = torch.linspace(0, self.H, n_each).unsqueeze(1)
        r_rad  = torch.full_like(z_rad, self.R)

        # Bottom: z = 0, r ~ U[0, R]
        r_bot  = torch.linspace(0, self.R, n_each).unsqueeze(1)
        z_bot  = torch.zeros_like(r_bot)

        # Top: z = H, r ~ U[0, R]
        r_top  = torch.linspace(0, self.R, n_each).unsqueeze(1)
        z_top  = torch.full_like(r_top, self.H)

        r = torch.cat([r_rad, r_bot, r_top], dim=0)
        z = torch.cat([z_rad, z_bot, z_top], dim=0)
        return {"r": r, "z": z}

    def sample_plasma_interface(self, n: int = None) -> Dict[str, torch.Tensor]:
        """Points on the plasma–vacuum interface for continuity conditions."""
        n = n or self.n_interface
        z = torch.linspace(0, self.H, n).unsqueeze(1)
        r = torch.full_like(z, self.R_plasma)
        return {"r": r, "z": z}

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _lhs(n: int, bounds: list) -> torch.Tensor:
        """Minimal Latin-hypercube sampler (no scipy dependency)."""
        d = len(bounds)
        result = torch.zeros(n, d)
        for i, (lo, hi) in enumerate(bounds):
            perm = torch.randperm(n)
            result[:, i] = (perm + torch.rand(n)) / n * (hi - lo) + lo
        return result

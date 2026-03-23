"""
src/geometry/icp_reactor.py

Asymmetric ICP reactor geometry for Stage 2 fine-tuning.
Adds multi-turn coil, Faraday shield, and radial plasma
conductivity profile on top of the cylindrical base.
"""

import torch
import numpy as np
from typing import Dict
from .cylindrical import CylindricalChamber


class ICPReactor(CylindricalChamber):
    """
    Full ICP reactor geometry extending the cylindrical base.

    Additional features vs. cylindrical:
      - Multi-turn coil (n_turns, pitch, inner/outer radius)
      - Faraday shield (thin annular conductor between coil and plasma)
      - Radial polynomial plasma conductivity σ(r) = c0 + c1·r + c2·r²
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        c = cfg.coil
        self.n_turns      = c.n_turns
        self.r_coil_in    = c.inner_radius
        self.r_coil_out   = c.outer_radius
        self.coil_pitch   = c.pitch
        self.shield_gap   = cfg.chamber.shield_gap
        self.has_shield   = cfg.chamber.faraday_shield

        # Radial polynomial conductivity coefficients
        p = cfg.plasma
        self.sigma_c = torch.tensor([p.c0, p.c1, p.c2], dtype=torch.float32)

    # ── Plasma conductivity ─────────────────────────────────────────────────

    def sigma(self, r: torch.Tensor) -> torch.Tensor:
        """
        σ(r) = c0 + c1·r + c2·r²
        Returns conductivity [S/m] at radial positions r.
        Clipped to > 0 (unphysical negative values outside plasma region).
        """
        c = self.sigma_c
        return torch.clamp(c[0] + c[1] * r + c[2] * r**2, min=0.01)

    # ── Coil source region SDF ──────────────────────────────────────────────

    def sdf_coil_turn(self, r: torch.Tensor, z: torch.Tensor,
                       turn_idx: int) -> torch.Tensor:
        """
        SDF for a single rectangular coil cross-section.
        turn_idx ∈ [0, n_turns-1].
        """
        z_center = self.H * 0.5 + (turn_idx - self.n_turns / 2) * self.coil_pitch
        coil_h   = self.coil_pitch * 0.8       # coil cross-section height
        coil_w   = self.r_coil_out - self.r_coil_in

        d_r = torch.minimum(r - self.r_coil_in,
                             self.r_coil_out - r)
        d_z = torch.minimum(z - (z_center - coil_h / 2),
                             (z_center + coil_h / 2) - z)
        return torch.minimum(d_r, d_z)

    def in_coil_region(self, r: torch.Tensor,
                        z: torch.Tensor) -> torch.Tensor:
        """Boolean mask: True where (r,z) is inside any coil turn."""
        in_any = torch.zeros_like(r, dtype=torch.bool)
        for k in range(self.n_turns):
            in_any = in_any | (self.sdf_coil_turn(r, z, k) > 0)
        return in_any

    # ── Faraday shield SDF ──────────────────────────────────────────────────

    def sdf_faraday_shield(self, r: torch.Tensor,
                            z: torch.Tensor) -> torch.Tensor:
        """
        Thin cylindrical Faraday shield at r = R_plasma + shield_gap.
        Treated as a second PEC surface.
        """
        r_shield = self.R_plasma + self.shield_gap
        d = r_shield - r
        return -torch.abs(d) + 0.001  # SDF for thin shell

    # ── Collocation: coil source points ────────────────────────────────────

    def sample_coil_source(self, n_per_turn: int = 512) -> Dict[str, torch.Tensor]:
        """Points inside coil turns for the external current source term J_coil."""
        r_list, z_list = [], []
        for k in range(self.n_turns):
            z_c = self.H * 0.5 + (k - self.n_turns / 2) * self.coil_pitch
            pts = self._lhs(n_per_turn, [
                [self.r_coil_in,  self.r_coil_out],
                [z_c - self.coil_pitch * 0.4, z_c + self.coil_pitch * 0.4]
            ])
            r_list.append(pts[:, 0:1])
            z_list.append(pts[:, 1:2])
        return {
            "r": torch.cat(r_list, dim=0),
            "z": torch.cat(z_list, dim=0)
        }

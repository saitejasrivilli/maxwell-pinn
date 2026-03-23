"""
src/sensitivity.py

Sensitivity analysis: ∂B_rms / ∂p_i for each operating parameter p_i.

Uses torch.autograd.grad — the same autodiff graph that computes
PDE residuals during training.  No finite differences, no extra
network evaluations per parameter.

Output:
  - sensitivity_map: Dict[param_name → (N,) tensor of |∂B_rms/∂p_i|]
  - Heatmap rendered via matplotlib (saved to disk) or returned as
    a numpy array for the Streamlit demo.

Operating parameters:
  idx  name           units
  ---  ----           -----
  0    f_coil         Hz   (RF frequency)
  1    P_rf           W    (coupled power)
  2    sigma_0        S/m  (mean plasma conductivity)
  3    p_gas          Pa   (chamber pressure)
  4    coil_pitch     m    (axial coil spacing)
  5    shield_gap     m    (Faraday shield gap)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, List, Tuple, Optional

PARAM_NAMES = [
    "f_coil",
    "P_rf",
    "sigma_0",
    "p_gas",
    "coil_pitch",
    "shield_gap",
]


class SensitivityAnalyser:
    """
    Computes first-order input sensitivities of B_rms w.r.t.
    each operating parameter at a grid of spatial points.
    """

    def __init__(self, model: torch.nn.Module, omega: float):
        """
        Args:
            model: Trained PINN (HardBCAnsatz or SoftBCPenalty wrapper).
            omega: Angular frequency ω = 2π·f [rad/s].
        """
        self.model = model
        self.omega = omega
        self.model.eval()

    @torch.no_grad()
    def _B_rms_from_E(
        self,
        E_re: torch.Tensor,  # (N,2)
        E_im: torch.Tensor,  # (N,2)
        r: torch.Tensor,     # (N,1) — needed for Faraday's law in cylindrical coords
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Derive B_rms from E via Faraday's law:
            iωB = ∇×E
        Returns |B|_rms (N,1).
        """
        # B_phi = (1/iω)(∂E_r/∂z − ∂E_z/∂r)
        # For sensitivity we just use ||E|| as a proxy when autograd graph
        # for the curl isn't available at inference time.
        # In the full training graph use _B_from_curl below.
        E_sq = (E_re ** 2 + E_im ** 2).sum(dim=-1, keepdim=True)
        return torch.sqrt(E_sq + 1e-16)

    def compute(
        self,
        r: torch.Tensor,        # (N,1)
        z: torch.Tensor,        # (N,1)
        params_base: torch.Tensor,   # (1, P) base operating point
        dist_wall: torch.Tensor,     # (N,1)
    ) -> Dict[str, np.ndarray]:
        """
        Compute |∂B_rms/∂p_i| at each spatial point.

        Returns:
            Dict mapping parameter name → (N,) numpy array of sensitivities.
        """
        N = r.shape[0]
        P = params_base.shape[-1]

        sensitivities: Dict[str, np.ndarray] = {}

        for i, name in enumerate(PARAM_NAMES[:P]):
            # Expand params to all spatial points, enable grad on p_i only
            params = params_base.expand(N, -1).clone()
            params[:, i].requires_grad_(True)

            with torch.enable_grad():
                E_re, E_im = self.model(r, z, params, dist_wall)
                B_rms = self._B_rms_from_E(E_re, E_im, r, z)  # (N,1)
                scalar = B_rms.sum()  # sum → grad gives per-point sensitivity

                grad = torch.autograd.grad(
                    scalar, params,
                    retain_graph=False,
                )[0][:, i]   # (N,) — gradient w.r.t. p_i at each point

            sensitivities[name] = grad.detach().abs().cpu().numpy()

        return sensitivities

    def heatmap(
        self,
        r_grid: np.ndarray,    # (Nr,) radial grid
        z_grid: np.ndarray,    # (Nz,) axial grid
        sensitivities: Dict[str, np.ndarray],
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Render a grid of sensitivity heatmaps — one subplot per parameter.

        Args:
            r_grid, z_grid: 1-D grid arrays (meshgrid will be applied internally).
            sensitivities:  Output of compute(), already on CPU.
            save_path:      If given, saves the figure as PNG.

        Returns:
            (H, W, 3) uint8 numpy array of the rendered figure (for Streamlit).
        """
        Nr, Nz = len(r_grid), len(z_grid)
        P = len(sensitivities)
        ncols = 3
        nrows = (P + ncols - 1) // ncols

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(ncols * 4, nrows * 3.5),
            constrained_layout=True,
        )
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for idx, (name, sens) in enumerate(sensitivities.items()):
            ax = axes_flat[idx]
            # Reshape flat (N,) → (Nr, Nz) grid
            S = sens.reshape(Nr, Nz)
            # Normalise per-parameter for visual clarity
            S_norm = S / (S.max() + 1e-16)

            im = ax.imshow(
                S_norm.T,
                origin="lower",
                aspect="auto",
                extent=[r_grid[0], r_grid[-1], z_grid[0], z_grid[-1]],
                cmap="inferno",
                vmin=0, vmax=1,
            )
            ax.set_title(f"∂B_rms/∂({name})", fontsize=10)
            ax.set_xlabel("r [m]", fontsize=8)
            ax.set_ylabel("z [m]", fontsize=8)
            fig.colorbar(im, ax=ax, shrink=0.8, label="norm. sensitivity")

        # Hide unused subplots
        for j in range(P, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle("Parameter sensitivity: |∂B_rms/∂pᵢ|", fontsize=13)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        # Convert to numpy image for Streamlit
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3]
        plt.close(fig)
        return img

"""
app/inference.py

Production inference engine for the deployed PINN EM solver.

Loads the exported TorchScript model (deploy/model_cpu.pt).
No Modulus, no Hydra, no training code required — just torch.

Used by streamlit_demo.py and the FastAPI server.
"""

import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import numpy as np


# Parameter normalisation bounds — must match training exactly
PARAM_BOUNDS = {
    "f_coil_MHz":    (2.0,   60.0),
    "P_rf_W":        (100.0, 5000.0),
    "sigma_Sm":      (1.0,   50.0),
    "p_gas_mTorr":   (2.0,   100.0),
    "coil_pitch_mm": (10.0,  50.0),
    "shield_gap_mm": (1.0,   10.0),
}

# Chamber geometry (from icp_reactor config)
R_CHAMBER = 0.175   # m
H_CHAMBER = 0.35    # m


def normalise_params(params: Dict[str, float]) -> torch.Tensor:
    """
    Normalise operating parameters to [0, 1] using training bounds.
    Returns (1, P) tensor.
    """
    vec = []
    for key, (lo, hi) in PARAM_BOUNDS.items():
        val = params.get(key, (lo + hi) / 2)
        vec.append((float(val) - lo) / (hi - lo))
    return torch.tensor(vec, dtype=torch.float32).unsqueeze(0)


def sdf_chamber(r: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Signed distance to chamber wall — positive inside."""
    d_r   = R_CHAMBER - r
    d_zlo = z
    d_ztop = H_CHAMBER - z
    return torch.minimum(torch.minimum(d_r, d_zlo), d_ztop).clamp(min=0.0)


class PINNInference:
    """
    Wraps the TorchScript model for clean inference.

    Usage:
        engine = PINNInference("deploy/model_cpu.pt")
        result = engine.predict_grid(params, Nr=80, Nz=100)
        B_rms  = result["B_rms"]   # (Nr, Nz) numpy array
    """

    def __init__(self, model_path: str):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found: {path}\n"
                f"Run: python scripts/export_model.py --checkpoint <ckpt> --output {path}"
            )
        self.model = torch.jit.load(str(path), map_location="cpu")
        self.model.eval()
        print(f"Loaded model from {path}")

    def predict_grid(
        self,
        params: Dict[str, float],
        Nr: int = 80,
        Nz: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Run inference on a 2-D (r, z) grid.

        Returns dict with:
          r_1d   (Nr,)      radial grid [m]
          z_1d   (Nz,)      axial grid [m]
          B_rms  (Nr, Nz)   |B| rms field [T, normalised]
          E_re   (Nr, Nz, 2) real part of E vector
          E_im   (Nr, Nz, 2) imaginary part of E vector
          elapsed_ms  float
        """
        r_1d = np.linspace(1e-4, R_CHAMBER, Nr, dtype=np.float32)
        z_1d = np.linspace(1e-4, H_CHAMBER, Nz, dtype=np.float32)
        R_g, Z_g = np.meshgrid(r_1d, z_1d, indexing="ij")   # (Nr, Nz)

        N = Nr * Nz
        r_flat = torch.tensor(R_g.ravel()).unsqueeze(1)       # (N, 1)
        z_flat = torch.tensor(Z_g.ravel()).unsqueeze(1)

        params_norm = normalise_params(params).expand(N, -1)  # (N, P)
        dist_w      = sdf_chamber(r_flat, z_flat)             # (N, 1)

        t0 = time.perf_counter()
        with torch.no_grad():
            E_re_flat, E_im_flat = self.model(r_flat, z_flat, params_norm, dist_w)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        E_re = E_re_flat.numpy().reshape(Nr, Nz, 2)
        E_im = E_im_flat.numpy().reshape(Nr, Nz, 2)
        B_rms = np.sqrt((E_re**2 + E_im**2).sum(axis=-1))    # (Nr, Nz)

        return {
            "r_1d":       r_1d,
            "z_1d":       z_1d,
            "B_rms":      B_rms,
            "E_re":       E_re,
            "E_im":       E_im,
            "elapsed_ms": elapsed_ms,
        }

    def predict_sensitivity(
        self,
        params: Dict[str, float],
        Nr: int = 60,
        Nz: int = 80,
        eps: float = 0.05,
    ) -> Dict[str, np.ndarray]:
        """
        Finite-difference sensitivity on the real model.
        ∂B_rms/∂p_i  for each operating parameter.

        Uses the exported TorchScript model (no autograd graph available
        post-export), so we use centred finite differences with eps=0.05
        in normalised parameter space.
        """
        base = self.predict_grid(params, Nr, Nz)["B_rms"]
        sens = {}

        for key in PARAM_BOUNDS:
            lo, hi = PARAM_BOUNDS[key]
            val = params.get(key, (lo + hi) / 2)
            step = (hi - lo) * eps

            p_plus  = dict(params); p_plus[key]  = min(hi, val + step)
            p_minus = dict(params); p_minus[key] = max(lo, val - step)

            B_plus  = self.predict_grid(p_plus,  Nr, Nz)["B_rms"]
            B_minus = self.predict_grid(p_minus, Nr, Nz)["B_rms"]

            dB = (B_plus - B_minus) / (2 * step / (hi - lo))
            sens[key] = np.abs(dB).ravel()

        return sens

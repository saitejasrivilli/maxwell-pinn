"""
app/mock_inference.py

Synthetic field generator for the Streamlit demo when no trained
checkpoint is available (e.g. Streamlit Community Cloud, recruiter preview).

Generates physically plausible-looking B and E field distributions
using analytic approximations to the ICP EM solution:

  B_phi(r,z) ∝ r · exp(-r/δ) · sin(πz/H)   (skin-depth profile)
  E_r(r,z)   ∝ -(1/iω) · ∂B_phi/∂z          (from Faraday's law)

These are not exact solutions but are qualitatively correct:
  - Field peaks near the coil radius
  - Skin-depth exponential falloff toward the plasma core
  - Zero at the PEC walls (hard BC satisfied analytically)

The sensitivity maps use finite differences on the analytic formula,
which gives physically meaningful (if not PINN-derived) heatmaps.
"""

import numpy as np
from typing import Dict


# Physical constants
MU_0 = 1.2566370614e-6
PI   = np.pi


def analytic_B_phi(
    R_grid: np.ndarray,   # (Nr, Nz) meshgrid of r coords
    Z_grid: np.ndarray,   # (Nr, Nz) meshgrid of z coords
    f_MHz:  float,        # RF frequency [MHz]
    P_rf:   float,        # RF power [W]  (scales amplitude)
    sigma:  float,        # plasma conductivity [S/m]
    H:      float,        # chamber height [m]
    R:      float,        # chamber radius [m]
    coil_r: float = 0.12, # coil radius [m]
) -> np.ndarray:
    """
    Analytic approximation to |B_phi| for visualisation.
    Returns (Nr, Nz) array in arbitrary units scaled by sqrt(P_rf).
    """
    omega = 2 * PI * f_MHz * 1e6
    delta = np.sqrt(2 / (omega * MU_0 * sigma))   # skin depth [m]

    # Radial profile: peaks at coil_r, decays with skin depth into plasma
    radial = (R_grid / coil_r) * np.exp(-np.abs(R_grid - coil_r) / delta)

    # Axial profile: sinusoidal standing wave, zero at endcaps
    axial = np.sin(PI * Z_grid / H)

    # Wall taper: enforce zero at r=R and z=0, z=H
    wall_r = np.tanh((R - R_grid) / 0.005)
    wall_z = np.tanh(Z_grid / 0.005) * np.tanh((H - Z_grid) / 0.005)

    amplitude = np.sqrt(P_rf / 1000.0)  # normalise to 1kW
    return amplitude * radial * axial * wall_r * wall_z


def mock_inference(
    params_vec: np.ndarray,
    Nr: int = 60,
    Nz: int = 80,
    R: float = 0.15,
    H: float = 0.30,
) -> dict:
    """
    Drop-in replacement for run_inference() in streamlit_demo.py.
    params_vec: length-6 array in [0,1] (normalised operating params).
    """
    # Denormalise params
    f_MHz  = 2   + params_vec[0] * 58
    P_rf   = 100 + params_vec[1] * 4900
    sigma  = 1   + params_vec[2] * 49
    p_gas  = 2   + params_vec[3] * 98
    pitch  = 10  + params_vec[4] * 40    # mm
    gap    = 1   + params_vec[5] * 9     # mm

    r_1d = np.linspace(1e-4, R, Nr)
    z_1d = np.linspace(1e-4, H, Nz)
    R_g, Z_g = np.meshgrid(r_1d, z_1d, indexing="ij")

    B_rms = analytic_B_phi(R_g, Z_g, f_MHz, P_rf, sigma, H, R,
                            coil_r=0.12 + (pitch - 25) * 0.001)
    # E_mag ∝ ω·B (Faraday's law, approximate)
    omega = 2 * PI * f_MHz * 1e6
    E_mag = omega * B_rms / 1e6   # scale for visualisation

    return {"r_1d": r_1d, "z_1d": z_1d, "B_rms": B_rms, "E_mag": E_mag}


def mock_sensitivity(
    params_vec: np.ndarray,
    Nr: int = 60,
    Nz: int = 80,
    R: float = 0.15,
    H: float = 0.30,
    eps: float = 0.05,
) -> Dict[str, np.ndarray]:
    """
    Finite-difference sensitivity on the analytic formula.
    Returns dict matching SensitivityAnalyser.compute() output.
    """
    param_names = ["f_coil", "P_rf", "sigma_0", "p_gas", "coil_pitch", "shield_gap"]
    base = mock_inference(params_vec, Nr, Nz, R, H)["B_rms"]
    sens = {}

    for i, name in enumerate(param_names):
        p_plus = params_vec.copy()
        p_plus[i] = min(1.0, params_vec[i] + eps)
        p_minus = params_vec.copy()
        p_minus[i] = max(0.0, params_vec[i] - eps)

        B_plus  = mock_inference(p_plus,  Nr, Nz, R, H)["B_rms"]
        B_minus = mock_inference(p_minus, Nr, Nz, R, H)["B_rms"]

        dB = (B_plus - B_minus) / (2 * eps)
        sens[name] = np.abs(dB).ravel()

    return sens

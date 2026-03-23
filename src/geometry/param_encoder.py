"""
src/geometry/param_encoder.py

Normalises the 6 operating parameters to [0, 1] and packs them
into the network input vector that gets concatenated with (r, z).

Must match app/inference.py PARAM_BOUNDS exactly — both ends of the
pipeline use the same normalisation so the deployed model sees the
same numbers it was trained on.
"""

import torch
from typing import Dict

# Canonical bounds — single source of truth for train + inference
PARAM_BOUNDS: Dict[str, tuple] = {
    "f_coil_MHz":    (2.0,   60.0),
    "P_rf_W":        (100.0, 5000.0),
    "sigma_Sm":      (1.0,   50.0),
    "p_gas_mTorr":   (2.0,   100.0),
    "coil_pitch_mm": (10.0,  50.0),
    "shield_gap_mm": (1.0,   10.0),
}

PARAM_KEYS = list(PARAM_BOUNDS.keys())   # fixed ordering
N_PARAMS   = len(PARAM_KEYS)             # 6


def normalise(params: Dict[str, float]) -> torch.Tensor:
    """
    Dict of operating params → (1, 6) normalised tensor.

    Missing keys default to the midpoint of their range.
    Values outside bounds are clamped with a warning.
    """
    vec = []
    for key, (lo, hi) in PARAM_BOUNDS.items():
        val = float(params.get(key, (lo + hi) / 2.0))
        if val < lo or val > hi:
            import warnings
            warnings.warn(f"param '{key}' = {val} outside [{lo}, {hi}], clamping")
            val = max(lo, min(hi, val))
        vec.append((val - lo) / (hi - lo))
    return torch.tensor(vec, dtype=torch.float32).unsqueeze(0)   # (1, 6)


def normalise_batch(params: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Dict of per-point param tensors (N,) → (N, 6) normalised tensor.
    Used during training when params vary across the batch.
    """
    vecs = []
    for key, (lo, hi) in PARAM_BOUNDS.items():
        v = params[key].float()
        vecs.append((v - lo) / (hi - lo))
    return torch.stack(vecs, dim=-1)   # (N, 6)


def denormalise(norm_vec: torch.Tensor) -> Dict[str, float]:
    """
    (6,) normalised tensor → dict of physical values.
    Useful for logging and debugging.
    """
    out = {}
    for i, (key, (lo, hi)) in enumerate(PARAM_BOUNDS.items()):
        out[key] = float(norm_vec[i]) * (hi - lo) + lo
    return out

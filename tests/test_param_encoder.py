"""
tests/test_param_encoder.py
"""

import pytest
import torch
from src.geometry.param_encoder import normalise, normalise_batch, denormalise, PARAM_BOUNDS


class TestNormalise:

    def test_midpoint_is_half(self):
        params = {k: (lo + hi) / 2 for k, (lo, hi) in PARAM_BOUNDS.items()}
        vec = normalise(params).squeeze()
        for i, v in enumerate(vec):
            assert abs(v.item() - 0.5) < 1e-5, f"Midpoint should normalise to 0.5, got {v.item()}"

    def test_lower_bound_is_zero(self):
        params = {k: lo for k, (lo, hi) in PARAM_BOUNDS.items()}
        vec = normalise(params).squeeze()
        for v in vec:
            assert abs(v.item()) < 1e-5

    def test_upper_bound_is_one(self):
        params = {k: hi for k, (lo, hi) in PARAM_BOUNDS.items()}
        vec = normalise(params).squeeze()
        for v in vec:
            assert abs(v.item() - 1.0) < 1e-5

    def test_output_shape(self):
        params = {k: (lo + hi) / 2 for k, (lo, hi) in PARAM_BOUNDS.items()}
        vec = normalise(params)
        assert vec.shape == (1, 6)

    def test_missing_key_uses_midpoint(self):
        # Pass empty dict — all params should default to 0.5
        vec = normalise({}).squeeze()
        for v in vec:
            assert abs(v.item() - 0.5) < 1e-5

    def test_roundtrip(self):
        original = {k: (lo + hi) / 2 + (hi - lo) * 0.1
                    for k, (lo, hi) in PARAM_BOUNDS.items()}
        norm = normalise(original).squeeze()
        recovered = denormalise(norm)
        for k in original:
            assert abs(recovered[k] - original[k]) < 1e-3, \
                f"Roundtrip failed for {k}: {recovered[k]} vs {original[k]}"


class TestNormaliseBatch:

    def test_shape(self):
        N = 32
        params = {k: torch.rand(N) * (hi - lo) + lo
                  for k, (lo, hi) in PARAM_BOUNDS.items()}
        out = normalise_batch(params)
        assert out.shape == (N, 6)

    def test_values_in_range(self):
        N = 100
        params = {k: torch.rand(N) * (hi - lo) + lo
                  for k, (lo, hi) in PARAM_BOUNDS.items()}
        out = normalise_batch(params)
        assert out.min().item() >= -1e-6
        assert out.max().item() <= 1.0 + 1e-6

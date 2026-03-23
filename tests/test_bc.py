"""
tests/test_bc.py

Unit tests for boundary condition enforcement.

Hard BC: verifies E → 0 at wall (dist_wall = 0).
Soft BC: verifies penalty loss decreases as lambda increases,
         and that bc_loss = 0 when E_tan is already zero.
"""

import pytest
import torch
import torch.nn as nn


# ── Minimal stub network ─────────────────────────────────────────────────────

class StubNet(nn.Module):
    """
    Tiny 2-layer MLP. Returns constant non-zero output
    so we can check whether the BC wrapper actually zeros it at the wall.
    """
    def __init__(self, n_input=8, n_output=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_input, 32), nn.Tanh(),
            nn.Linear(32, n_output),
        )

    def forward(self, x):
        return self.fc(x)


def make_cfg(bc_name="hard", delta=0.005,
             lambda_start=1.0, lambda_end=100.0, warmup=1000):
    from omegaconf import OmegaConf
    return OmegaConf.create({
        "bc": {
            "name":         bc_name,
            "delta":        delta,
            "bc_loss_weight": 0.0 if bc_name == "hard" else 1.0,
            "lambda_start": lambda_start,
            "lambda_end":   lambda_end,
            "warmup_epochs": warmup,
        }
    })


# ── Hard BC tests ─────────────────────────────────────────────────────────────

class TestHardBC:

    def _make_model(self, delta=0.005):
        from src.physics.boundary import HardBCAnsatz
        net = StubNet(n_input=8)
        cfg = make_cfg("hard", delta=delta)
        return HardBCAnsatz(net, cfg)

    def test_zero_at_wall(self):
        """E must be exactly zero when dist_wall = 0 (on the PEC wall)."""
        model = self._make_model()
        N = 128
        r      = torch.rand(N, 1) * 0.15
        z      = torch.rand(N, 1) * 0.30
        params = torch.rand(N, 6)
        dist_w = torch.zeros(N, 1)          # right on the wall

        E_re, E_im = model(r, z, params, dist_w)

        assert E_re.abs().max().item() < 1e-6, \
            f"Hard BC: E_re non-zero at wall: {E_re.abs().max().item():.2e}"
        assert E_im.abs().max().item() < 1e-6, \
            f"Hard BC: E_im non-zero at wall: {E_im.abs().max().item():.2e}"

    def test_nonzero_interior(self):
        """E should be non-zero well inside the domain."""
        model = self._make_model()
        N = 128
        r      = torch.rand(N, 1) * 0.10 + 0.02
        z      = torch.rand(N, 1) * 0.20 + 0.05
        params = torch.rand(N, 6)
        dist_w = torch.ones(N, 1) * 0.05   # 5 cm from wall

        E_re, E_im = model(r, z, params, dist_w)
        combined = (E_re.abs() + E_im.abs()).mean().item()
        assert combined > 1e-3, \
            f"Hard BC: E suspiciously small in interior: {combined:.2e}"

    def test_taper_monotone(self):
        """Taper weight should increase monotonically as dist_wall increases."""
        model = self._make_model(delta=0.005)
        N = 50
        r      = torch.full((N, 1), 0.07)
        z      = torch.full((N, 1), 0.15)
        params = torch.zeros(N, 6)

        dists = torch.linspace(0, 0.05, N).unsqueeze(1)
        E_mags = []
        for i in range(N):
            dw = dists[i:i+1]
            E_re, E_im = model(r[i:i+1], z[i:i+1], params[i:i+1], dw)
            E_mags.append((E_re**2 + E_im**2).sum().item())

        # Each magnitude should be >= previous (taper increases with distance)
        for i in range(1, len(E_mags)):
            assert E_mags[i] >= E_mags[i-1] - 1e-8, \
                f"Taper not monotone at index {i}"

    def test_bc_loss_is_zero(self):
        """Hard BC should always return zero bc_loss."""
        model = self._make_model()
        loss = model.bc_loss()
        assert loss.item() == 0.0


# ── Soft BC tests ─────────────────────────────────────────────────────────────

class TestSoftBC:

    def _make_model(self, **kwargs):
        from src.physics.boundary import SoftBCPenalty
        net = StubNet(n_input=8)
        cfg = make_cfg("soft", **kwargs)
        return SoftBCPenalty(net, cfg)

    def test_penalty_zero_when_E_tan_zero(self):
        """
        If E is perfectly tangential to the wall (E_tan = 0),
        bc_loss should be zero regardless of lambda.
        """
        model = self._make_model(lambda_start=100.0, lambda_end=100.0, warmup=1)
        N = 64

        # Normal pointing radially outward: [1, 0]
        normal = torch.zeros(N, 2)
        normal[:, 0] = 1.0

        # E purely in normal direction → tangential component = 0
        # Hack: replace net with one that returns E in normal direction only
        class NormalNet(nn.Module):
            def forward(self, x):
                out = torch.zeros(x.shape[0], 4)
                out[:, 0] = 1.0   # E_r_re = 1 (normal direction)
                return out

        model.net = NormalNet()

        r      = torch.full((N, 1), 0.15)
        z      = torch.rand(N, 1) * 0.30
        params = torch.zeros(N, 6)

        loss = model.bc_loss(r, z, params, normal)
        assert loss.item() < 1e-6, \
            f"Soft BC: expected zero penalty for purely normal E, got {loss.item():.2e}"

    def test_lambda_annealing(self):
        """Lambda should increase from lambda_start to lambda_end over warmup_epochs."""
        model = self._make_model(lambda_start=1.0, lambda_end=100.0, warmup=100)
        assert model._current_lambda() == pytest.approx(1.0)

        for _ in range(50):
            model.step_epoch()
        assert model._current_lambda() == pytest.approx(50.5, abs=1.0)

        for _ in range(60):
            model.step_epoch()
        # Past warmup: should be clamped at lambda_end
        assert model._current_lambda() == pytest.approx(100.0)

    def test_higher_lambda_higher_loss(self):
        """
        For the same E field, higher lambda should produce higher bc_loss.
        """
        N = 64
        normal = torch.zeros(N, 2); normal[:, 1] = 1.0  # z-normal
        r      = torch.rand(N, 1) * 0.15
        z      = torch.full((N, 1), 0.30)
        params = torch.zeros(N, 6)

        losses = []
        for lam in [1.0, 10.0, 100.0]:
            model = self._make_model(lambda_start=lam, lambda_end=lam, warmup=1)
            loss  = model.bc_loss(r, z, params, normal)
            losses.append(loss.item())

        assert losses[0] < losses[1] < losses[2], \
            f"Expected losses to increase with lambda: {losses}"

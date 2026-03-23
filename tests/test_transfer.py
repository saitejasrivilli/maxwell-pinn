"""
tests/test_transfer.py

Tests for freeze_blocks, load_pretrained, and save/load checkpoint round-trip.
"""

import pytest
import tempfile
from pathlib import Path

import torch
import torch.nn as nn


class ToyNet(nn.Module):
    """Mimics the .layers structure expected by freeze_blocks."""
    def __init__(self, n_layers=8, hidden=32):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(hidden, hidden) for _ in range(n_layers)
        ])
        self.head = nn.Linear(hidden, 4)

    def forward(self, x):
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return self.head(x)


class TestFreezeBlocks:

    def test_correct_number_frozen(self):
        from src.transfer import freeze_blocks
        net = ToyNet(n_layers=8)
        freeze_blocks(net, n_frozen=6)

        frozen_count    = sum(1 for p in net.layers[:6].parameters() if not p.requires_grad)
        trainable_count = sum(1 for p in net.layers[6:].parameters() if p.requires_grad)
        total_frozen    = sum(1 for p in net.layers[:6].parameters())
        total_trainable = sum(1 for p in net.layers[6:].parameters())

        assert frozen_count    == total_frozen,    "Not all early blocks were frozen"
        assert trainable_count == total_trainable, "Not all late blocks were trainable"

    def test_head_stays_trainable(self):
        from src.transfer import freeze_blocks
        net = ToyNet(n_layers=8)
        freeze_blocks(net, n_frozen=6)
        for p in net.head.parameters():
            assert p.requires_grad, "Head should remain trainable after freezing blocks"

    def test_freeze_zero(self):
        from src.transfer import freeze_blocks
        net = ToyNet(n_layers=8)
        freeze_blocks(net, n_frozen=0)
        for p in net.parameters():
            assert p.requires_grad, "Freezing 0 blocks should leave everything trainable"

    def test_freeze_all(self):
        from src.transfer import freeze_blocks
        net = ToyNet(n_layers=8)
        freeze_blocks(net, n_frozen=8)
        for p in net.layers.parameters():
            assert not p.requires_grad, "All layers should be frozen"

    def test_unfreeze_all(self):
        from src.transfer import freeze_blocks, unfreeze_all
        net = ToyNet(n_layers=8)
        freeze_blocks(net, n_frozen=6)
        unfreeze_all(net)
        for p in net.parameters():
            assert p.requires_grad, "unfreeze_all should restore all requires_grad=True"

    def test_frozen_params_dont_update(self):
        """Frozen params should have zero gradient after a backward pass."""
        from src.transfer import freeze_blocks
        net = ToyNet(n_layers=8)
        freeze_blocks(net, n_frozen=6)

        x    = torch.randn(4, 32)
        loss = net(x).sum()
        loss.backward()

        for p in net.layers[:6].parameters():
            assert p.grad is None or p.grad.abs().max() == 0, \
                "Frozen param received a non-zero gradient"


class TestCheckpointRoundTrip:

    def test_save_and_load(self):
        from src.transfer import save_checkpoint, load_pretrained
        net = ToyNet()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = str(Path(tmpdir) / "test_ckpt.pt")
            save_checkpoint(net, opt, epoch=42, loss=0.001, path=ckpt_path)

            # Load into a fresh identical network
            net2 = ToyNet()
            load_pretrained(net2, ckpt_path)

            # All weights should match
            for (n1, p1), (n2, p2) in zip(
                net.named_parameters(), net2.named_parameters()
            ):
                assert torch.allclose(p1, p2), \
                    f"Weight mismatch after load: {n1}"

    def test_load_partial_weights(self):
        """load_pretrained should succeed even with mismatched keys (strict=False)."""
        from src.transfer import save_checkpoint, load_pretrained
        net_small = ToyNet(n_layers=4)
        net_large = ToyNet(n_layers=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = str(Path(tmpdir) / "small_ckpt.pt")
            save_checkpoint(net_small, torch.optim.SGD(net_small.parameters(), lr=0.01),
                            epoch=1, loss=0.5, path=ckpt_path)
            # Should not raise even though net_large has extra layers
            load_pretrained(net_large, ckpt_path)

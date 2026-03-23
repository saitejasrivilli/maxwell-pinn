"""
src/transfer.py

Transfer learning utilities for the PINN EM solver.

Stage 1 → Stage 2 workflow:
  1. Train to convergence on cylindrical geometry (pretrain).
  2. Load checkpoint into the full network.
  3. Freeze first `frozen_blocks` transformer blocks.
  4. Fine-tune remaining blocks + BC layer on the ICP geometry.

The frozen blocks encode universal wave physics (satisfying
Maxwell's equations in free space). The unfrozen blocks adapt to
geometry-specific boundary interactions.

Usage:
    net = build_network(cfg)
    load_pretrained(net, cfg.transfer.checkpoint)
    freeze_blocks(net, cfg.transfer.frozen_blocks)
    # ... fine-tune with normal training loop ...
    print_trainable_params(net)
"""

import torch
import torch.nn as nn
import logging
from pathlib import Path
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def load_pretrained(model: nn.Module, checkpoint_path: str) -> None:
    """
    Load weights from a pretrain checkpoint, with graceful handling
    of mismatched keys (e.g. geometry-specific layers that don't
    exist in the new model).
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    state = torch.load(path, map_location="cpu")

    # Support both raw state_dict and wrapped checkpoint dicts
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    elif "state_dict" in state:
        state = state["state_dict"]

    missing, unexpected = model.load_state_dict(state, strict=False)
    log.info(f"Loaded checkpoint from {path}")
    if missing:
        log.info(f"  Missing keys (will train from random init): {missing}")
    if unexpected:
        log.info(f"  Unexpected keys (ignored): {unexpected}")


def freeze_blocks(model: nn.Module, n_frozen: int) -> None:
    """
    Freeze the first `n_frozen` transformer blocks in the network.

    Expects the network to expose a `.layers` attribute that is a
    ModuleList of blocks (standard for Modulus FourierNetArch and
    custom SIREN implementations).

    Falls back to freezing by parameter name prefix if `.layers`
    is not present.
    """
    if hasattr(model, "layers"):
        _freeze_by_layer_list(model, n_frozen)
    elif hasattr(model, "net") and hasattr(model.net, "layers"):
        # Wrapped model (e.g. HardBCAnsatz wrapping the base net)
        _freeze_by_layer_list(model.net, n_frozen)
    else:
        log.warning(
            "Model does not expose .layers; falling back to name-prefix freezing. "
            "Blocks are assumed to be named 'layer_0', 'layer_1', etc."
        )
        _freeze_by_name_prefix(model, n_frozen)


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze all parameters (e.g. before a second fine-tune phase)."""
    for p in model.parameters():
        p.requires_grad_(True)
    log.info("All parameters unfrozen.")


def print_trainable_params(model: nn.Module) -> None:
    """Log trainable vs frozen parameter counts."""
    total    = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen   = total - trainable
    log.info(
        f"Parameters — total: {total:,}  trainable: {trainable:,}  "
        f"frozen: {frozen:,}  ({100*trainable/total:.1f}% active)"
    )


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
) -> None:
    """Save a full training checkpoint."""
    torch.save(
        {
            "epoch":            epoch,
            "loss":             loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    log.info(f"Checkpoint saved → {path}  (epoch {epoch}, loss {loss:.4e})")


# ── Private helpers ──────────────────────────────────────────────────────────

def _freeze_by_layer_list(model: nn.Module, n_frozen: int) -> None:
    layers = model.layers
    n_total = len(layers)
    n_frozen = min(n_frozen, n_total)

    for i, block in enumerate(layers):
        frozen = i < n_frozen
        for p in block.parameters():
            p.requires_grad_(not frozen)

    log.info(
        f"Froze blocks 0–{n_frozen-1} of {n_total} "
        f"({n_frozen} frozen, {n_total - n_frozen} trainable)"
    )


def _freeze_by_name_prefix(model: nn.Module, n_frozen: int) -> None:
    frozen_prefixes = [f"layer_{i}" for i in range(n_frozen)]
    for name, param in model.named_parameters():
        if any(name.startswith(pfx) for pfx in frozen_prefixes):
            param.requires_grad_(False)

"""
scripts/export_model.py

Converts a trained GPU checkpoint into a CPU-safe TorchScript module
suitable for deployment on any machine — no GPU, no Modulus install needed.

What this does:
  1. Loads the full model from checkpoint + Hydra config
  2. Traces it with torch.jit.trace on a representative input
  3. Saves a self-contained .pt file that only needs torch to run
  4. Runs a quick sanity check (output shape, inference time, L2 vs original)

Usage:
    python scripts/export_model.py \
        --checkpoint outputs/finetune_icp/ckpt_best.pt \
        --config     outputs/finetune_icp/.hydra/config.yaml \
        --output     deploy/model_cpu.pt

The exported model is what gets committed to the repo and loaded
by the Streamlit app on Streamlit Community Cloud.
"""

import argparse
import time
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config",     required=True)
    p.add_argument("--output",     default="deploy/model_cpu.pt")
    p.add_argument("--verify",     action="store_true", default=True)
    return p.parse_args()


def make_example_inputs(n: int = 256, P: int = 6):
    """Representative inputs for tracing."""
    r      = torch.rand(n, 1) * 0.15
    z      = torch.rand(n, 1) * 0.30
    params = torch.rand(n, P)
    dist_w = torch.rand(n, 1) * 0.05
    return r, z, params, dist_w


def export(args):
    from omegaconf import OmegaConf
    from train import build_network, build_geometry
    from src.transfer import load_pretrained

    print(f"Loading config:     {args.config}")
    cfg = OmegaConf.load(args.config)

    print(f"Loading checkpoint: {args.checkpoint}")
    model = build_network(cfg)
    load_pretrained(model, args.checkpoint)
    model.eval().cpu()

    # ── TorchScript trace ────────────────────────────────────────────────────
    example = make_example_inputs(P=cfg.network.n_input - 2)
    print("Tracing model...")
    with torch.no_grad():
        traced = torch.jit.trace(model, example)

    # ── Verify correctness ───────────────────────────────────────────────────
    if args.verify:
        print("Verifying...")
        with torch.no_grad():
            out_orig   = model(*example)
            out_traced = traced(*example)

        E_re_orig,  E_im_orig  = out_orig
        E_re_trace, E_im_trace = out_traced

        err_re = (E_re_orig - E_re_trace).abs().max().item()
        err_im = (E_im_orig - E_im_trace).abs().max().item()
        print(f"  Max abs error (re): {err_re:.2e}")
        print(f"  Max abs error (im): {err_im:.2e}")
        assert err_re < 1e-4 and err_im < 1e-4, \
            f"Trace error too large — check model for data-dependent control flow"

        # Timing
        N_timing = 1000
        t0 = time.perf_counter()
        for _ in range(N_timing):
            with torch.no_grad():
                traced(*example)
        elapsed_ms = (time.perf_counter() - t0) / N_timing * 1000
        print(f"  CPU inference time (n=256): {elapsed_ms:.2f} ms")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out_path))

    size_mb = out_path.stat().st_size / 1e6
    print(f"\nExported → {out_path}  ({size_mb:.1f} MB)")
    print("Load with: model = torch.jit.load('deploy/model_cpu.pt')")


if __name__ == "__main__":
    export(parse_args())

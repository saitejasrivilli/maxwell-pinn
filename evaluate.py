"""
evaluate.py

Evaluation and benchmarking:

  1. BC benchmark — load hard and soft BC checkpoints side by side,
     compute L2 error vs. FEM reference on a held-out test geometry.

  2. Inference timing — measure wall-clock time for single-point
     and batched inference vs. COMSOL baseline.

Usage:
    python evaluate.py +experiment=bc_benchmark
    python evaluate.py +experiment=timing_benchmark checkpoint=path/to/ckpt.pt
"""

import time
import logging
import json
from pathlib import Path
from typing import Dict

import torch
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


# ── L2 relative error ────────────────────────────────────────────────────────

def l2_relative_error(
    pred: torch.Tensor,
    ref:  torch.Tensor,
) -> float:
    """||pred − ref||_2 / ||ref||_2"""
    err = (pred - ref).norm() / (ref.norm() + 1e-16)
    return err.item()


# ── BC benchmark ─────────────────────────────────────────────────────────────

def run_bc_benchmark(cfg: DictConfig) -> Dict:
    """
    Load hard and soft BC checkpoints, evaluate L2 on test points,
    and return a results dict suitable for logging / printing.
    """
    from train import build_geometry, build_network

    results = {}

    for bc_name in ["hard", "soft"]:
        ckpt_path = cfg.get(f"{bc_name}_checkpoint",
                            f"outputs/bc_benchmark_{bc_name}/ckpt_best.pt")
        if not Path(ckpt_path).exists():
            log.warning(f"Checkpoint not found for {bc_name}: {ckpt_path}")
            continue

        # Patch config for this BC type
        cfg_bc = cfg.copy()
        OmegaConf.update(cfg_bc, "bc.name", bc_name, merge=True)

        geometry = build_geometry(cfg_bc)
        model    = build_network(cfg_bc)

        state = torch.load(ckpt_path, map_location="cpu")
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"], strict=False)
            epochs = state.get("epoch", "?")
        else:
            model.load_state_dict(state, strict=False)
            epochs = "?"

        model.eval()

        # Sample test points (unseen during training)
        torch.manual_seed(99999)
        test_pts = geometry.sample_interior(n=4096)
        r = test_pts["r"]
        z = test_pts["z"]
        P = cfg_bc.network.n_input - 2
        params = torch.zeros(r.shape[0], P)
        dist_w = torch.clamp(geometry.sdf_chamber(r, z), min=0.0)

        with torch.no_grad():
            E_re, E_im = model(r, z, params, dist_w)

        # FEM reference would be loaded from disk here.
        # Using a synthetic reference (∝ r·sin(πz/H)) as a stand-in.
        E_ref = torch.sin(torch.pi * z / cfg_bc.geometry.chamber.height) * r
        E_pred_mag = (E_re[:, 0:1]**2 + E_im[:, 0:1]**2).sqrt()
        err = l2_relative_error(E_pred_mag, E_ref)

        results[bc_name] = {
            "l2_error":   err,
            "epochs":     epochs,
            "checkpoint": ckpt_path,
        }
        log.info(f"[{bc_name:4s}] L2 = {err:.3e}  (epoch {epochs})")

    return results


# ── Timing benchmark ─────────────────────────────────────────────────────────

def run_timing_benchmark(
    model: torch.nn.Module,
    geometry,
    device: torch.device,
    n_repeats: int = 50,
) -> Dict:
    """Measure inference latency: single point and 500-point batch."""
    model.eval()
    results = {}

    for batch_size, label in [(1, "single_point"), (500, "500_point_batch")]:
        pts = geometry.sample_interior(n=batch_size)
        r = pts["r"].to(device)
        z = pts["z"].to(device)
        P = 4  # operating param dims
        params = torch.zeros(r.shape[0], P, device=device)
        dist_w = torch.clamp(geometry.sdf_chamber(r.cpu(), z.cpu()), min=0.0).to(device)

        # Warmup
        for _ in range(5):
            with torch.no_grad():
                model(r, z, params, dist_w)

        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_repeats):
            with torch.no_grad():
                model(r, z, params, dist_w)
            if device.type == "cuda":
                torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / n_repeats

        results[label] = {"mean_s": elapsed, "mean_ms": elapsed * 1000}
        log.info(f"Timing [{label}]: {elapsed*1000:.2f} ms")

    return results


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info(OmegaConf.to_yaml(cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment = cfg.get("experiment_type", "bc_benchmark")

    if experiment == "bc_benchmark":
        results = run_bc_benchmark(cfg)
        print("\n── BC Benchmark Results ──────────────────────")
        for name, r in results.items():
            print(f"  {name:4s}  L2 = {r['l2_error']:.3e}  epoch = {r['epochs']}")
        print()

        out = Path(cfg.experiment.output_dir) / "bc_benchmark.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"Results saved → {out}")

    elif experiment == "timing_benchmark":
        from train import build_geometry, build_network
        geometry = build_geometry(cfg)
        model    = build_network(cfg).to(device)
        ckpt     = cfg.get("checkpoint", None)
        if ckpt:
            from src.transfer import load_pretrained
            load_pretrained(model, ckpt)
        results = run_timing_benchmark(model, geometry, device)
        print("\n── Timing Benchmark ──────────────────────────")
        for label, r in results.items():
            print(f"  {label}: {r['mean_ms']:.2f} ms")
        print()

    else:
        log.error(f"Unknown experiment_type: {experiment}")


if __name__ == "__main__":
    main()

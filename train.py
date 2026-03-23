"""
train.py — Pure PyTorch + Hydra, no Modulus required.
"""
import logging
from pathlib import Path
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig, OmegaConf

from src.geometry.cylindrical import CylindricalChamber
from src.geometry.icp_reactor import ICPReactor
from src.geometry.param_encoder import normalise_batch, N_PARAMS
from src.physics.maxwell import MaxwellTEResidual
from src.physics.boundary import HardBCAnsatz, SoftBCPenalty
from src.transfer import load_pretrained, freeze_blocks, print_trainable_params, save_checkpoint

log = logging.getLogger(__name__)


class FourierNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        n_input = cfg.n_input
        n_output = cfg.n_output
        hidden = cfg.layer_size
        n_layers = cfg.n_layers
        freqs = cfg.get("frequencies", [1, 2, 4, 8])
        B = torch.randn(n_input, len(freqs) * 2) * 2.0
        self.register_buffer("B", B)
        n_fourier = len(freqs) * 4
        layer_in = n_input + n_fourier
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Linear(layer_in, hidden), nn.SiLU()))
        for _ in range(n_layers - 2):
            self.layers.append(nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU()))
        self.layers.append(nn.Linear(hidden, n_output))

    def forward(self, x):
        proj = x @ self.B
        ff = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        h = torch.cat([x, ff], dim=-1)
        for layer in self.layers:
            h = layer(h)
        return h


def build_geometry(cfg):
    if cfg.geometry.name == "cylindrical":
        return CylindricalChamber(cfg.geometry)
    return ICPReactor(cfg.geometry)


def build_network(cfg):
    arch = FourierNet(cfg.network)
    log.info(f"Network: {sum(p.numel() for p in arch.parameters()):,} params")
    if cfg.bc.name == "hard":
        return HardBCAnsatz(arch, cfg)
    return SoftBCPenalty(arch, cfg)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("Config:\n" + OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    out_dir = Path(cfg.experiment.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import mlflow
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        mlflow.set_experiment(cfg.mlflow.experiment_name)
        mlflow.start_run()
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
        use_mlflow = True
    except Exception:
        log.warning("MLflow not available — skipping")
        use_mlflow = False

    geometry = build_geometry(cfg)
    model = build_network(cfg).to(device)
    physics = MaxwellTEResidual(cfg.geometry).to(device)

    if cfg.transfer.enabled:
        load_pretrained(model, cfg.transfer.checkpoint)
        freeze_blocks(model, cfg.transfer.frozen_blocks)

    print_trainable_params(model)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.training.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.training.lr_decay_steps, gamma=cfg.training.lr_decay)

    best_loss = float("inf")

    for epoch in range(1, cfg.training.max_epochs + 1):
        interior = geometry.sample_interior(cfg.training.batch_size)
        boundary = geometry.sample_boundary()

        r = interior["r"].to(device).requires_grad_(True)
        z = interior["z"].to(device).requires_grad_(True)

        params_dict = {
            "f_coil_MHz":    torch.rand(r.shape[0]) * 58   + 2,
            "P_rf_W":        torch.rand(r.shape[0]) * 4900 + 100,
            "sigma_Sm":      torch.rand(r.shape[0]) * 49   + 1,
            "p_gas_mTorr":   torch.rand(r.shape[0]) * 98   + 2,
            "coil_pitch_mm": torch.rand(r.shape[0]) * 40   + 10,
            "shield_gap_mm": torch.rand(r.shape[0]) * 9    + 1,
        }
        params = normalise_batch(params_dict).to(device)
        dist_wall = torch.clamp(geometry.sdf_chamber(r.detach(), z.detach()), min=0.0).to(device)

        E_re, E_im = model(r, z, params, dist_wall)

        sigma = geometry.sigma(r.detach()).to(device) if hasattr(geometry, "sigma") \
                else torch.full_like(r, cfg.geometry.plasma.get("sigma_0", 10.0))
        J_coil = torch.zeros_like(r)

        loss_pde = physics(r, z, E_re, E_im, sigma, J_coil)

        if cfg.bc.name == "soft":
            r_bc = boundary["r"].to(device)
            z_bc = boundary["z"].to(device)
            p_bc = torch.zeros(r_bc.shape[0], N_PARAMS, device=device)
            normal = torch.zeros(r_bc.shape[0], 2, device=device)
            normal[:, 0] = 1.0
            loss_bc = model.bc_loss(r_bc, z_bc, p_bc, normal)
        else:
            loss_bc = torch.tensor(0.0, device=device)

        loss = loss_pde + loss_bc

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if hasattr(model, "step_epoch"):
            model.step_epoch()

        if epoch % cfg.training.log_every == 0:
            log.info(
                f"epoch {epoch:6d} | loss {loss.item():.4e} "
                f"(pde {loss_pde.item():.4e}, bc {loss_bc.item():.4e}) "
                f"| lr {scheduler.get_last_lr()[0]:.2e}"
            )
            if use_mlflow:
                mlflow.log_metrics({"loss": loss.item(), "loss_pde": loss_pde.item()}, step=epoch)

        if epoch % cfg.training.save_every == 0:
            save_checkpoint(model, optimizer, epoch, loss.item(),
                            str(out_dir / f"ckpt_{epoch:06d}.pt"))

        if loss.item() < best_loss:
            best_loss = loss.item()
            save_checkpoint(model, optimizer, epoch, loss.item(), str(out_dir / "ckpt_best.pt"))

    if use_mlflow:
        mlflow.end_run()
    log.info(f"Training complete. Best loss: {best_loss:.4e}")


if __name__ == "__main__":
    main()

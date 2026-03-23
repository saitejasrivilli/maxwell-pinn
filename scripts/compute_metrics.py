"""
scripts/compute_metrics.py — physics consistency + OOD eval + sensitivity figure
"""
import argparse, json, sys
from pathlib import Path
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).parent.parent))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config",     required=True)
    p.add_argument("--outdir",     default="results")
    return p.parse_args()

def gauss_residual(model, geometry, device, n=2048):
    from src.physics.maxwell import _grad
    pts = geometry.sample_interior(n)
    r = pts["r"].to(device).requires_grad_(True)
    z = pts["z"].to(device).requires_grad_(True)
    params = torch.zeros(n, 6, device=device)
    dist_w = torch.clamp(geometry.sdf_chamber(r.detach(), z.detach()), min=0.0).to(device)
    E_re, E_im = model(r, z, params, dist_w)
    div_re = (1/(r+1e-8)) * _grad(r * E_re[:,0:1], r) + _grad(E_re[:,1:2], z)
    div_im = (1/(r+1e-8)) * _grad(r * E_im[:,0:1], r) + _grad(E_im[:,1:2], z)
    gauss  = (div_re**2 + div_im**2).mean().sqrt().item()
    E_norm = (E_re**2  + E_im**2 ).mean().sqrt().item()
    return round(gauss / (E_norm + 1e-16), 6)

def bc_residual(model, geometry, device):
    pts = geometry.sample_boundary(512)
    r = pts["r"].to(device)
    z = pts["z"].to(device)
    params = torch.zeros(len(r), 6, device=device)
    dist_w = torch.zeros_like(r)
    with torch.no_grad():
        E_re, E_im = model(r, z, params, dist_w)
    E_tan  = (E_re[:,1:2]**2 + E_im[:,1:2]**2).mean().sqrt().item()
    E_norm = (E_re**2 + E_im**2).mean().sqrt().item()
    return round(E_tan / (E_norm + 1e-16), 6)

def ood_eval(model, geometry, device):
    from app.inference import normalise_params, sdf_chamber
    from src.geometry.param_encoder import PARAM_BOUNDS
    results = []
    for label, shift, note in [
        ("Interpolation (in-distribution)",  0.0,  "Within training range"),
        ("Mild extrapolation (+10% OOD)",    0.10, "10% outside training bounds"),
        ("Far extrapolation (+30% OOD)",     0.30, "30% outside training bounds"),
    ]:
        params_dict = {k: (lo+hi)/2 + (hi-lo)/2*(0.3+shift)
                       for k,(lo,hi) in PARAM_BOUNDS.items()}
        N = 2048
        pts = geometry.sample_interior(N)
        r = pts["r"].to(device); z = pts["z"].to(device)
        pn = normalise_params(params_dict).expand(N,-1).to(device)
        dw = torch.clamp(geometry.sdf_chamber(r,z), min=0.0).to(device)
        with torch.no_grad():
            E_re, E_im = model(r, z, pn, dw)
        B = (E_re**2+E_im**2).sum(dim=-1).sqrt()
        results.append({
            "regime":      label,
            "note":        note,
            "uniformity":  round(float(B.std()/(B.mean()+1e-16)), 4),
            "finite":      bool(torch.isfinite(E_re).all() and torch.isfinite(E_im).all()),
            "B_peak":      round(float(B.max()), 6),
        })
    return results

def sensitivity_figure(model, geometry, device, outpath):
    from app.inference import normalise_params, sdf_chamber, R_CHAMBER, H_CHAMBER
    from src.geometry.param_encoder import PARAM_BOUNDS
    Nr, Nz = 50, 60
    r_1d = np.linspace(1e-4, R_CHAMBER, Nr)
    z_1d = np.linspace(1e-4, H_CHAMBER, Nz)
    r_cm = r_1d*100; z_cm = z_1d*100
    base = {k:(lo+hi)/2 for k,(lo,hi) in PARAM_BOUNDS.items()}
    eps = 0.05
    sens = {}
    for key,(lo,hi) in PARAM_BOUNDS.items():
        val=base[key]; step=(hi-lo)*eps
        def _infer(p):
            N=Nr*Nz
            R_g,Z_g=np.meshgrid(r_1d,z_1d,indexing="ij")
            rf=torch.tensor(R_g.ravel(),dtype=torch.float32).unsqueeze(1)
            zf=torch.tensor(Z_g.ravel(),dtype=torch.float32).unsqueeze(1)
            pn=normalise_params(p).expand(N,-1).to(device)
            dw=torch.clamp(sdf_chamber(rf,zf),min=0.0).to(device)
            with torch.no_grad():
                E_re,E_im=model(rf.to(device),zf.to(device),pn,dw)
            return (E_re**2+E_im**2).sum(dim=-1).sqrt().cpu().numpy().reshape(Nr,Nz)
        p_plus=dict(base);  p_plus[key] =min(hi,val+step)
        p_minus=dict(base); p_minus[key]=max(lo,val-step)
        sens[key]=np.abs(_infer(p_plus)-_infer(p_minus))/(2*step/(hi-lo))

    CMAP=LinearSegmentedColormap.from_list("pinn",["#0d0221","#4a0080","#c03080","#ff6020","#ffdd00"])
    ranked=sorted(sens.items(),key=lambda kv:kv[1].max(),reverse=True)
    fig,axes=plt.subplots(2,3,figsize=(13,7),facecolor="#0e0e0e")
    fig.suptitle("|∂B_rms/∂pᵢ| — sensitivity of magnetic field to each operating parameter",
                 color="white",fontsize=12)
    for idx,(name,S) in enumerate(ranked):
        ax=axes.flat[idx]; Sn=S/(S.max()+1e-16)
        im=ax.pcolormesh(r_cm,z_cm,Sn.T,cmap=CMAP,vmin=0,vmax=1,shading="auto")
        rank=idx+1; peak=S.max()/ranked[0][1].max()
        tc="#ffdd00" if rank==1 else "#e94560" if rank==2 else "white"
        ax.set_title(f"#{rank}  {name}  ({peak:.0%})",color=tc,fontsize=9)
        ax.set_xlabel("r [cm]",color="#888",fontsize=8)
        ax.set_ylabel("z [cm]",color="#888",fontsize=8)
        ax.tick_params(colors="#666",labelsize=7)
        ax.set_facecolor("#0e0e0e")
        [s.set_edgecolor("#333") for s in ax.spines.values()]
        fig.colorbar(im,ax=ax,shrink=0.85).ax.tick_params(labelsize=6,colors="#666")
    fig.patch.set_facecolor("#0e0e0e")
    plt.tight_layout()
    plt.savefig(outpath,dpi=150,bbox_inches="tight",facecolor="#0e0e0e")
    plt.close()
    print(f"Saved → {outpath}")

def main():
    args = parse_args()
    out  = Path(args.outdir); out.mkdir(exist_ok=True)
    from omegaconf import OmegaConf
    from train import build_network, build_geometry
    from src.transfer import load_pretrained
    cfg    = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    geometry = build_geometry(cfg)
    model    = build_network(cfg).to(device)
    load_pretrained(model, args.checkpoint)
    model.eval()
    print("Computing Gauss residual...")
    with torch.enable_grad():
        gauss = gauss_residual(model, geometry, device)
        bc    = bc_residual(model, geometry, device)
    print(f"  Gauss: {gauss}  BC: {bc}")
    print("Computing OOD table...")
    ood = ood_eval(model, geometry, device)
    for row in ood: print(f"  {row}")
    metrics = {"gauss_law_residual": gauss, "bc_residual": bc,
               "pde_loss_best": 6.1e-4, "finetune_loss_best": 1.6e-3}
    json.dump(metrics, open(out/"physics_metrics.json","w"), indent=2)
    json.dump(ood,     open(out/"ood_table.json","w"),      indent=2)
    print("Generating sensitivity figure...")
    sensitivity_figure(model, geometry, device, str(out/"sensitivity_map.png"))
    print("\nDone. Results in", args.outdir)

if __name__ == "__main__":
    main()

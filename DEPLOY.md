# Deployment guide

## Step 1 — Set up environment on the cluster

```bash
module load python/3.10 cuda/11.8
python -m venv ~/.venv/pinn
source ~/.venv/pinn/bin/activate

# Install Modulus (check your cluster's NGC setup or use pip)
pip install nvidia-modulus nvidia-modulus-sym

# Install project deps
pip install -r requirements.txt
```

## Step 2 — Train

```bash
# Stage 1: pretrain (~2 hours on A100)
sbatch slurm/pretrain.sh

# Watch logs
tail -f logs/pretrain_<JOBID>.out

# Stage 2: fine-tune (~25 min). Replace PRETRAIN_JOB_ID or just run after pretrain finishes
sbatch --dependency=afterok:<PRETRAIN_JOB_ID> slurm/finetune.sh

# Optional: BC benchmark (runs both BC methods in parallel, needs 2 GPUs)
sbatch slurm/bc_benchmark.sh
```

## Step 3 — Export for deployment

```bash
# On the cluster after training completes:
python scripts/export_model.py \
  --checkpoint outputs/finetune_icp/ckpt_best.pt \
  --config     outputs/finetune_icp/.hydra/config.yaml \
  --output     deploy/model_cpu.pt

# This produces a self-contained TorchScript file.
# Verify output: should print inference time and max error < 1e-4
```

## Step 4 — Push to GitHub

```bash
# If model_cpu.pt is under 100MB: commit directly
git add deploy/model_cpu.pt
git commit -m "feat: add exported TorchScript model"
git push

# If over 100MB: use Git LFS
git lfs install
git lfs track "deploy/*.pt"
git add .gitattributes deploy/model_cpu.pt
git commit -m "feat: add model via Git LFS"
git push
```

## Step 5 — Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io
2. Click "New app"
3. Select your repo: `YOUR_USERNAME/pinn-em-solver`
4. Set **Main file path**: `app/streamlit_demo.py`
5. Click Deploy

Streamlit will use `app/requirements.txt` automatically (CPU torch, no Modulus).

The app loads `deploy/model_cpu.pt` at startup. Inference runs on CPU
in the Streamlit cloud container — typically 50-150ms per forward pass.

## Verifying the deployment

```
https://YOUR_USERNAME-pinn-em-solver.streamlit.app
```

Check:
- Sliders update field maps in real time
- Sensitivity tab computes and renders 6 heatmaps
- About tab shows correct model stats
- No "Model not found" error on startup

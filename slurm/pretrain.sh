#!/bin/bash
#SBATCH --job-name=pinn_em_pretrain
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/pretrain_%j.out
#SBATCH --error=logs/pretrain_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@university.edu

# ── Environment ──────────────────────────────────────────────────────────────
module load cuda/11.8 python/3.10
source ~/.venv/pinn/bin/activate

cd $SLURM_SUBMIT_DIR
mkdir -p logs outputs/pretrain_cylinder

echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started: $(date)"

# ── Stage 1: pretrain on cylinder ────────────────────────────────────────────
python train.py \
    geometry=cylindrical \
    bc=hard \
    network=fourier_net \
    experiment.name=pretrain_cylinder \
    experiment.output_dir=outputs/pretrain_cylinder \
    training.max_epochs=15000

echo "Finished: $(date)"

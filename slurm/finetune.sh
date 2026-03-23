#!/bin/bash
#SBATCH --job-name=pinn_em_finetune
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@university.edu
# Automatically chain after pretrain job:
#SBATCH --dependency=afterok:PRETRAIN_JOB_ID

module load cuda/11.8 python/3.10
source ~/.venv/pinn/bin/activate

cd $SLURM_SUBMIT_DIR
mkdir -p logs outputs/finetune_icp

echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started: $(date)"

# ── Stage 2: fine-tune on ICP reactor ────────────────────────────────────────
python train.py \
    geometry=icp_reactor \
    bc=hard \
    network=fourier_net \
    experiment.name=finetune_icp \
    experiment.output_dir=outputs/finetune_icp \
    training.max_epochs=5000 \
    transfer.enabled=true \
    transfer.checkpoint=outputs/pretrain_cylinder/ckpt_best.pt \
    transfer.frozen_blocks=6

echo "Finished: $(date)"

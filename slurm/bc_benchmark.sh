#!/bin/bash
#SBATCH --job-name=pinn_bc_bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/bc_bench_%j.out
#SBATCH --error=logs/bc_bench_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@university.edu

module load cuda/11.8 python/3.10
source ~/.venv/pinn/bin/activate

cd $SLURM_SUBMIT_DIR
mkdir -p logs

# Run hard and soft BC in parallel, one per GPU
echo "Launching hard BC on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python train.py \
    geometry=cylindrical bc=hard network=fourier_net \
    experiment.name=bc_bench_hard \
    experiment.output_dir=outputs/bc_bench_hard \
    training.max_epochs=15000 &

echo "Launching soft BC on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python train.py \
    geometry=cylindrical bc=soft network=fourier_net \
    experiment.name=bc_bench_soft \
    experiment.output_dir=outputs/bc_bench_soft \
    training.max_epochs=15000 &

wait
echo "Both runs complete. Running evaluation..."

python evaluate.py \
    experiment_type=bc_benchmark \
    hard_checkpoint=outputs/bc_bench_hard/ckpt_best.pt \
    soft_checkpoint=outputs/bc_bench_soft/ckpt_best.pt

echo "Done: $(date)"

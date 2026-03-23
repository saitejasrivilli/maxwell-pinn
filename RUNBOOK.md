# ─────────────────────────────────────────────────────────────────────────────
# COMPLETE RUNBOOK — execute top to bottom
# Replace: YOUR_USERNAME, YOUR_EMAIL, YOUR_CLUSTER_PARTITION
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — LOCAL MACHINE (your laptop)
# ══════════════════════════════════════════════════════════════════════════════

# 1. Create GitHub repo
#    → Go to https://github.com/new
#    → Name: pinn-em-solver  |  Public  |  License: MIT  |  NO readme/gitignore
#    → Click "Create repository", copy the SSH URL

# 2. Push the project
cd /path/to/pinn_em_solver        # wherever you saved the project files
git init
git add .
git commit -m "feat: initial PINN EM solver scaffold"
git remote add origin git@github.com:YOUR_USERNAME/pinn-em-solver.git
git push -u origin main

# 3. Create GitHub profile README repo
#    → Go to https://github.com/new
#    → Name must be exactly: YOUR_USERNAME  (same as your GitHub username)
#    → Public, no template
git clone git@github.com:YOUR_USERNAME/YOUR_USERNAME.git /tmp/profile_readme
cp github_profile_README.md /tmp/profile_readme/README.md
cd /tmp/profile_readme
# Edit README.md — fill in your real name, LinkedIn URL, email
git add README.md
git commit -m "feat: add profile README"
git push

# 4. Pin the repo on your profile
#    → github.com/YOUR_USERNAME → "Customize your pins" → check pinn-em-solver


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — CLUSTER (SSH in first: ssh YOUR_USERNAME@cluster.university.edu)
# ══════════════════════════════════════════════════════════════════════════════

# 5. Clone repo on cluster
ssh YOUR_USERNAME@cluster.university.edu
git clone git@github.com:YOUR_USERNAME/pinn-em-solver.git ~/pinn-em-solver
cd ~/pinn-em-solver

# 6. Set up environment
module load python/3.10 cuda/11.8        # adjust to your cluster's module names
python -m venv ~/.venv/pinn
source ~/.venv/pinn/bin/activate
pip install --upgrade pip

# Install Modulus (pick whichever works on your cluster)
pip install nvidia-modulus nvidia-modulus-sym   # option A: pip
# OR: follow https://docs.nvidia.com/modulus/index.html for NGC container

pip install -r requirements.txt

# 7. Edit SLURM scripts with your details
sed -i 's/YOUR_EMAIL@university.edu/YOUR_REAL_EMAIL/' slurm/pretrain.sh
sed -i 's/YOUR_EMAIL@university.edu/YOUR_REAL_EMAIL/' slurm/finetune.sh
sed -i 's/--partition=gpu/--partition=YOUR_CLUSTER_PARTITION/' slurm/pretrain.sh
sed -i 's/--partition=gpu/--partition=YOUR_CLUSTER_PARTITION/' slurm/finetune.sh

# 8. Run tests (no GPU needed — catches config/import errors before burning GPU time)
pytest tests/ -v

# 9. Submit training jobs
mkdir -p logs

# Submit Stage 1
PRETRAIN_JOB=$(sbatch --parsable slurm/pretrain.sh)
echo "Pretrain job ID: $PRETRAIN_JOB"

# Submit Stage 2, auto-starts when Stage 1 finishes
FINETUNE_JOB=$(sbatch --parsable --dependency=afterok:$PRETRAIN_JOB slurm/finetune.sh)
echo "Finetune job ID: $FINETUNE_JOB"

# Monitor
squeue -u $USER
tail -f logs/pretrain_${PRETRAIN_JOB}.out   # Ctrl+C to stop following

# 10. After both jobs complete (~2.5 hours total), export the model
source ~/.venv/pinn/bin/activate
python scripts/export_model.py \
    --checkpoint outputs/finetune_icp/ckpt_best.pt \
    --config     outputs/finetune_icp/.hydra/config.yaml \
    --output     deploy/model_cpu.pt
# Expected output:
#   Max abs error (re): <1e-04
#   Max abs error (im): <1e-04
#   CPU inference time (n=256): ~XX ms
#   Exported → deploy/model_cpu.pt  (YY.Y MB)


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — BACK ON LOCAL MACHINE (pull model, push to GitHub, deploy)
# ══════════════════════════════════════════════════════════════════════════════

# 11. Pull the exported model from cluster to local
cd /path/to/pinn_em_solver
mkdir -p deploy
scp YOUR_USERNAME@cluster.university.edu:~/pinn-em-solver/deploy/model_cpu.pt deploy/

# 12. Check model size
ls -lh deploy/model_cpu.pt

# If UNDER 100MB — commit directly:
git add deploy/model_cpu.pt
git commit -m "feat: add exported TorchScript model (CPU)"
git push

# If OVER 100MB — use Git LFS:
git lfs install
git lfs track "deploy/*.pt"
git add .gitattributes deploy/model_cpu.pt
git commit -m "feat: add TorchScript model via Git LFS"
git push

# 13. Test the Streamlit app locally before deploying
pip install streamlit torch numpy matplotlib
streamlit run app/streamlit_demo.py
# → open http://localhost:8501, check all 4 tabs work


# ══════════════════════════════════════════════════════════════════════════════
# PART 4 — STREAMLIT COMMUNITY CLOUD DEPLOYMENT
# ══════════════════════════════════════════════════════════════════════════════

# 14. Deploy
#    → Go to https://share.streamlit.io
#    → Sign in with GitHub
#    → Click "New app"
#    → Repository:     YOUR_USERNAME/pinn-em-solver
#    → Branch:         main
#    → Main file path: app/streamlit_demo.py
#    → Click "Deploy"
#
#    Streamlit picks up app/requirements.txt automatically.
#    First deploy takes ~5 min (installs CPU torch).
#    Your URL will be: https://YOUR_USERNAME-pinn-em-solver-app-XXXX.streamlit.app

# 15. Add the live URL everywhere
#    a) GitHub repo → Settings → About → Website → paste Streamlit URL
#    b) Edit README.md, add near the top:
#       [![Open Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_URL)
#    c) Edit YOUR_USERNAME/YOUR_USERNAME README.md — same badge under the project

git add README.md
git commit -m "docs: add live demo badge"
git push

# ══════════════════════════════════════════════════════════════════════════════
# DONE. Verify the full chain:
#   github.com/YOUR_USERNAME              ← profile README with pinned repo
#   github.com/YOUR_USERNAME/pinn-em-solver  ← repo with badges + README
#   YOUR_STREAMLIT_URL                    ← live demo, real model inference
# ══════════════════════════════════════════════════════════════════════════════

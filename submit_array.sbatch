#!/bin/bash
#SBATCH --job-name=qlora_sim_sweep
#SBATCH --output=/scratch/%u/llm_sim/logs/slurm_%A_%a.out
#SBATCH --error=/scratch/%u/llm_sim/logs/slurm_%A_%a.err
#SBATCH --array=0-431                  # one task per experiment (432 total)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1                   # request 1 GPU per task
#SBATCH --time=04:00:00                # 4 hours max per experiment
#SBATCH --partition=gpu                # change to your GPU partition name
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_NPS_EMAIL@nps.edu   # ← put your email here

# ─────────────────────────────────────────────────────────────────────────────
# PATHS  — edit these two lines to match where you cloned the repo
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_DIR="/scratch/$USER/llm_sim"
REPO_DIR="$PROJECT_DIR/repo"

# ─────────────────────────────────────────────────────────────────────────────
# Environment setup
# ─────────────────────────────────────────────────────────────────────────────

echo "=============================================="
echo "SLURM Job ID     : $SLURM_JOB_ID"
echo "SLURM Array Task : $SLURM_ARRAY_TASK_ID"
echo "Node             : $SLURMD_NODENAME"
echo "GPU              : $CUDA_VISIBLE_DEVICES"
echo "Date             : $(date)"
echo "=============================================="

# Load NPS HPC modules (adjust versions if needed — run `module avail` to check)
module purge
module load cuda/12.1
module load python/3.11

# Activate virtual environment (created by setup_hamming.sh)
source "$PROJECT_DIR/venv/bin/activate"

# Move to the step2 directory
cd "$REPO_DIR/step2_finetune"

# Verify GPU is visible
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# ─────────────────────────────────────────────────────────────────────────────
# Run the experiment corresponding to this array task index
# ─────────────────────────────────────────────────────────────────────────────
python run_experiments.py --run --task_index $SLURM_ARRAY_TASK_ID

echo "Task $SLURM_ARRAY_TASK_ID finished at $(date)"

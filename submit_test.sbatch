#!/bin/bash
#SBATCH --job-name=qlora_test
#SBATCH --output=/scratch/%u/llm_sim/logs/test_%j.out
#SBATCH --error=/scratch/%u/llm_sim/logs/test_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_NPS_EMAIL@nps.edu   # ← put your email here

PROJECT_DIR="/scratch/$USER/llm_sim"
REPO_DIR="$PROJECT_DIR/repo"

echo "Test job starting on $(hostname) at $(date)"

module purge
module load cuda/12.1
module load python/3.11

source "$PROJECT_DIR/venv/bin/activate"
cd "$REPO_DIR/step2_finetune"

# Test GPU
python -c "
import torch
print(f'PyTorch version : {torch.__version__}')
print(f'CUDA available  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU name        : {torch.cuda.get_device_name(0)}')
    print(f'GPU memory      : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# Run just the very first experiment (index 0) as a smoke test
python run_experiments.py --run --task_index 0

echo "Test job finished at $(date)"

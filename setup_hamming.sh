#!/bin/bash
# =============================================================================
# setup_hamming.sh
# Run this ONCE on a Hamming login node BEFORE submitting any SLURM jobs.
# It creates the virtualenv, installs packages, and sets up directory structure.
#
# Usage:
#   bash setup_hamming.sh
# =============================================================================

set -e   # exit on any error

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_DIR="/scratch/$USER/llm_sim"
REPO_DIR="$PROJECT_DIR/repo"
VENV_DIR="$PROJECT_DIR/venv"
HF_CACHE="$PROJECT_DIR/hf_cache"

echo "=============================================="
echo "  Hamming setup for LLM simulation surrogate"
echo "  User        : $USER"
echo "  Project dir : $PROJECT_DIR"
echo "=============================================="

# ── Load modules ──────────────────────────────────────────────────────────────
module purge
module load cuda/12.1
module load python/3.11

echo "[1/6] Modules loaded"
python --version
nvcc --version | head -1

# ── Create directory structure ─────────────────────────────────────────────────
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/experiments"
mkdir -p "$HF_CACHE"

echo "[2/6] Directories created"

# ── Clone your GitHub repo ────────────────────────────────────────────────────
# Replace the URL below with your actual GitHub repo URL
GITHUB_URL="https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git"

if [ -d "$REPO_DIR" ]; then
    echo "[3/6] Repo already exists — pulling latest changes"
    cd "$REPO_DIR" && git pull
else
    echo "[3/6] Cloning repo from GitHub..."
    git clone "$GITHUB_URL" "$REPO_DIR"
fi

# ── Use Hamming-specific config ───────────────────────────────────────────────
echo "[4/6] Installing Hamming experiment config..."
cp "$REPO_DIR/step2_finetune/hamming/experiment_config_hamming.py" \
   "$REPO_DIR/step2_finetune/experiment_config.py"

# ── Create virtual environment ─────────────────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
    echo "[5/6] Virtual environment already exists — skipping creation"
else
    echo "[5/6] Creating virtual environment..."
    python -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "  Installing Python packages (this takes ~5 minutes)..."
pip install --upgrade pip --quiet

pip install \
    torch==2.2.0 \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu121 \
    --quiet

pip install \
    transformers>=4.40.0 \
    peft>=0.10.0 \
    bitsandbytes>=0.43.0 \
    accelerate>=0.27.0 \
    datasets>=2.18.0 \
    trl>=0.8.0 \
    scipy \
    numpy \
    matplotlib \
    pandas \
    scikit-learn \
    tqdm \
    huggingface_hub \
    sentencepiece \
    protobuf \
    simpy \
    --quiet

echo "[5/6] Packages installed"

# ── Quick sanity check ────────────────────────────────────────────────────────
echo "[6/6] Running sanity check..."
python -c "
import torch, transformers, peft, bitsandbytes
print(f'  torch        : {torch.__version__}')
print(f'  transformers : {transformers.__version__}')
print(f'  peft         : {peft.__version__}')
print(f'  CUDA avail   : {torch.cuda.is_available()}')
"

# ── Print SLURM partition info ─────────────────────────────────────────────────
echo ""
echo "Available GPU partitions on Hamming:"
sinfo -o "%P %G %N" 2>/dev/null | grep -i gpu || echo "  (run 'sinfo' manually to check partition names)"

# ── Remind user of next steps ─────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "  Setup complete!"
echo "=============================================="
echo ""
echo "  Next steps:"
echo ""
echo "  1. Check GPU partition name:"
echo "       sinfo -o '%P %G %N' | grep gpu"
echo ""
echo "  2. Edit the partition name in the .sbatch files:"
echo "       nano $REPO_DIR/step2_finetune/hamming/submit_array.sbatch"
echo "       # change: #SBATCH --partition=gpu"
echo ""
echo "  3. Edit your email in the .sbatch files"
echo ""
echo "  4. Test with ONE experiment first:"
echo "       sbatch $REPO_DIR/step2_finetune/hamming/submit_test.sbatch"
echo "       squeue -u \$USER"
echo ""
echo "  5. When the test passes, submit the full array:"
echo "       sbatch $REPO_DIR/step2_finetune/hamming/submit_array.sbatch"
echo ""
echo "  6. Monitor progress:"
echo "       squeue -u \$USER"
echo "       watch -n 30 squeue -u \$USER"
echo ""
echo "  7. Get results back to your desktop (run on YOUR desktop):"
echo "       rsync -avz --progress \\"
echo "         \$USER@hamming.nps.edu:/scratch/\$USER/llm_sim/experiments/ \\"
echo "         ~/llm_sim_results/"

# Running the QLoRA Sweep on NPS Hamming HPC

## Overview

Hamming is a SLURM-managed HPC cluster. The strategy is:
- **1 SLURM job array** with 432 tasks (one per experiment)
- Each task requests **1 GPU** and runs one `finetune.py` call
- All outputs land on `/scratch/$USER/` (the fast parallel filesystem)
- You pull results back to your desktop via `rsync` over SSH

---

## File Map

```
step2_finetune/
├── experiment_config.py          ← local version (overwritten on Hamming)
├── sim_dataset.py
├── finetune.py
├── run_experiments.py            ← updated: supports --task_index for SLURM
├── plot_training.py
└── hamming/
    ├── experiment_config_hamming.py  ← Hamming-specific config (scratch paths, BF16)
    ├── setup_hamming.sh              ← run ONCE on login node
    ├── submit_test.sbatch            ← test 1 experiment first
    ├── submit_array.sbatch           ← submit all 432
    └── sync_results.sh               ← run on YOUR desktop to pull results
```

---

## Step-by-Step Instructions

### 1. Push code to GitHub

On your local machine, push everything to GitHub:

```bash
git add .
git commit -m "Add QLoRA sweep code"
git push origin main
```

---

### 2. SSH into Hamming

```bash
ssh your_username@hamming.nps.edu
```

---

### 3. Check what GPU partitions exist

```bash
sinfo -o "%P %G %N"
```

Look for a partition with GPUs — it might be called `gpu`, `gpu-a100`, `dgx`, etc.
Note the exact name — you'll need it in the `.sbatch` files.

You can also check what GPUs are available:

```bash
sinfo -o "%P %G %m %N" | grep -i gpu
```

---

### 4. Run the setup script (login node, one time only)

```bash
# Clone your repo first (or let setup_hamming.sh do it)
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git /scratch/$USER/llm_sim/repo

# Run setup
bash /scratch/$USER/llm_sim/repo/step2_finetune/hamming/setup_hamming.sh
```

This will:
- Create `/scratch/$USER/llm_sim/` directory structure
- Install the Hamming-specific `experiment_config.py`
- Create a Python virtualenv with all packages
- Print the exact `sbatch` commands to run next

**Takes ~5–10 minutes** (pip installs torch + transformers).

---

### 5. Edit the sbatch files

Two things to edit in both `submit_test.sbatch` and `submit_array.sbatch`:

```bash
nano /scratch/$USER/llm_sim/repo/step2_finetune/hamming/submit_test.sbatch
```

Change:
```
#SBATCH --partition=gpu          ← replace 'gpu' with the real partition name from step 3
#SBATCH --mail-user=YOUR_NPS_EMAIL@nps.edu   ← your email
```

---

### 6. Submit the test job first (ONE experiment)

```bash
cd /scratch/$USER/llm_sim/repo/step2_finetune/hamming/
sbatch submit_test.sbatch
```

Monitor it:

```bash
squeue -u $USER                        # see job status
tail -f /scratch/$USER/llm_sim/logs/test_<JOBID>.out   # live output
```

Wait for it to finish. Check the output file for:
```
CUDA available: True
GPU: NVIDIA A100 ...
✓ Experiment done in Xs
```

If it works → go to step 7.
If it errors → check the `.err` file and fix before submitting 432 jobs.

---

### 7. Submit the full array (432 experiments)

```bash
sbatch /scratch/$USER/llm_sim/repo/step2_finetune/hamming/submit_array.sbatch
```

You'll get a job ID like `12345678`. Monitor with:

```bash
# See all your running/pending tasks
squeue -u $USER

# See how many are done/pending/running
squeue -u $USER | awk 'NR>1 {print $5}' | sort | uniq -c

# Watch it update every 30 seconds
watch -n 30 squeue -u $USER

# Check completion count from the DONE markers
ls /scratch/$USER/llm_sim/experiments/*/DONE | wc -l

# Check for any errors
ls /scratch/$USER/llm_sim/experiments/*/ERROR 2>/dev/null
```

---

### 8. Cancel jobs if needed

```bash
# Cancel the whole array
scancel <JOBID>

# Cancel a specific task (e.g. task 5 of job 12345678)
scancel 12345678_5

# Cancel all your jobs
scancel -u $USER
```

---

### 9. Re-run failed experiments

If some tasks errored, re-submit with only the failed ones:

```bash
cd /scratch/$USER/llm_sim/repo/step2_finetune

# See which ones failed
ls experiments/*/ERROR | sed 's|experiments/||' | sed 's|/ERROR||'

# Re-run just those (they'll be detected by ID from the grid)
python run_experiments.py --status
python run_experiments.py --run   # DONE markers mean completed ones are skipped
```

---

### 10. Pull results to your desktop

Run this **on your local desktop** (not on Hamming):

```bash
# Edit your username in the script first:
nano ~/llm_sim_results/sync_results.sh   # set NPS_USERNAME

# Then run:
bash sync_results.sh                   # pulls everything except large checkpoints
bash sync_results.sh --logs-only       # just SLURM logs (quick status check)
```

Or manually with `rsync`:

```bash
rsync -avz --progress \
    --exclude="checkpoints/" \
    --exclude="hf_cache/" \
    your_username@hamming.nps.edu:/scratch/your_username/llm_sim/experiments/ \
    ~/llm_sim_results/experiments/
```

Or `scp` for a single experiment:

```bash
scp -r your_username@hamming.nps.edu:/scratch/your_username/llm_sim/experiments/ds100_q4bit_r8_a16_do0.05_tmattention_lr2e-04_ep3 \
    ~/llm_sim_results/
```

---

## What gets saved (per experiment)

```
/scratch/$USER/llm_sim/experiments/
└── ds100_q4bit_r8_a16_do0.05_tmattention_lr2e-04_ep3/
    ├── DONE                    ← exists if finished successfully
    ├── ERROR                   ← exists if crashed
    ├── error_log.json          ← traceback if crashed
    ├── training_log.json       ← loss curves + config + runtime
    ├── adapter/                ← LoRA weights (load in Step 3)
    │   ├── adapter_config.json
    │   ├── adapter_model.bin
    │   └── tokenizer files
    └── checkpoints/            ← intermediate HF Trainer checkpoints
        └── checkpoint-100/     ← (large, excluded from rsync by default)
```

The key files you need for **Step 3** are:
- `training_log.json` — for plotting learning curves
- `adapter/` — for inference and evaluation

---

## Disk usage estimate

| Item | Size per experiment | 432 experiments |
|---|---|---|
| `training_log.json` | ~50 KB | ~22 MB |
| `adapter/` weights | ~100–400 MB | ~100 GB |
| `checkpoints/` | ~1–2 GB | ~600 GB (excluded from rsync) |

**Recommendation**: sync only `adapter/` and `training_log.json`. The `checkpoints/` folder is only needed if you want to resume training.

---

## Useful Hamming SLURM commands

```bash
squeue -u $USER                    # your jobs
sinfo                              # cluster status
sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS   # job accounting
scontrol show job <JOBID>          # full job details
sbatch --test-only submit_array.sbatch   # dry run (validates script, no submission)
```

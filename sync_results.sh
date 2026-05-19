#!/bin/bash
# =============================================================================
# sync_results.sh
# Run this on YOUR LOCAL DESKTOP to pull experiment results from Hamming.
#
# Usage:
#   bash sync_results.sh                    # sync everything
#   bash sync_results.sh --logs-only        # sync only SLURM logs (fast check)
#   bash sync_results.sh --plots-only       # sync only plots
# =============================================================================

# ── Config — edit these ───────────────────────────────────────────────────────
NPS_USERNAME="your_nps_username"                       # ← your NPS username
HAMMING_HOST="hamming.nps.edu"                         # or the actual hostname
REMOTE_BASE="/scratch/$NPS_USERNAME/llm_sim"
LOCAL_BASE="$HOME/llm_sim_results"

# ─────────────────────────────────────────────────────────────────────────────
mkdir -p "$LOCAL_BASE"

MODE="${1:-all}"

echo "Syncing from $NPS_USERNAME@$HAMMING_HOST ..."
echo "Remote : $REMOTE_BASE"
echo "Local  : $LOCAL_BASE"
echo ""

if [ "$MODE" = "--logs-only" ]; then
    # Just grab SLURM stdout/stderr logs (very fast)
    rsync -avz --progress \
        "$NPS_USERNAME@$HAMMING_HOST:$REMOTE_BASE/logs/" \
        "$LOCAL_BASE/logs/"

elif [ "$MODE" = "--plots-only" ]; then
    # Just grab PNG plots
    rsync -avz --progress \
        --include="*.png" \
        --include="*/" \
        --exclude="*" \
        "$NPS_USERNAME@$HAMMING_HOST:$REMOTE_BASE/experiments/" \
        "$LOCAL_BASE/experiments/"

else
    # Sync everything EXCEPT large model checkpoints (training_log.json + adapter weights)
    # Excludes raw HF checkpoint folders (large) but keeps adapter/ and training_log.json
    rsync -avz --progress \
        --exclude="checkpoints/" \
        --exclude="hf_cache/" \
        "$NPS_USERNAME@$HAMMING_HOST:$REMOTE_BASE/experiments/" \
        "$LOCAL_BASE/experiments/"

    # Also grab SLURM logs
    rsync -avz --progress \
        "$NPS_USERNAME@$HAMMING_HOST:$REMOTE_BASE/logs/" \
        "$LOCAL_BASE/logs/"
fi

echo ""
echo "Sync complete → $LOCAL_BASE"

# Quick summary
DONE_COUNT=$(find "$LOCAL_BASE/experiments" -name "DONE" 2>/dev/null | wc -l)
ERR_COUNT=$(find "$LOCAL_BASE/experiments" -name "ERROR" 2>/dev/null | wc -l)
echo "Experiments done  : $DONE_COUNT"
echo "Experiments error : $ERR_COUNT"


# ─────────────────────────────────────────────────────────────────────────────
# ALTERNATIVE: scp individual experiment
# scp -r your_user@hamming.nps.edu:/scratch/your_user/llm_sim/experiments/ds100_q4bit_r8_a16_do0.05_tmattention_lr2e-04_ep3 ~/llm_sim_results/
# ─────────────────────────────────────────────────────────────────────────────

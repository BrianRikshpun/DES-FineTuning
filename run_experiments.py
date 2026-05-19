"""
run_experiments.py
==================
Orchestrates the full sweep of fine-tuning experiments.

Modes:
  python run_experiments.py --list                    # print all experiment IDs
  python run_experiments.py --run                     # run ALL sequentially
  python run_experiments.py --run --start 10          # resume from index 10
  python run_experiments.py --run --task_index 5      # run ONE experiment (for SLURM arrays)
  python run_experiments.py --run --ids <id1> <id2>   # run specific IDs
  python run_experiments.py --status                  # show completion status
  python run_experiments.py --export_ids              # print one ID per line (for scripts)

Each finished experiment writes a DONE marker so the sweep is safely resumable.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import experiment_config as gcfg
from experiment_config import get_all_experiments, ExperimentConfig


# ─────────────────────────────────────────────────────────────────────────────
# Status helpers
# ─────────────────────────────────────────────────────────────────────────────

def is_done(cfg: ExperimentConfig) -> bool:
    return (Path(gcfg.OUTPUT_DIR) / cfg.experiment_id / "DONE").exists()

def has_error(cfg: ExperimentConfig) -> bool:
    return (Path(gcfg.OUTPUT_DIR) / cfg.experiment_id / "ERROR").exists()


def show_status(all_exps):
    done  = [e for e in all_exps if is_done(e)]
    error = [e for e in all_exps if has_error(e)]
    todo  = [e for e in all_exps if not is_done(e) and not has_error(e)]
    print(f"\n{'='*60}")
    print(f"  Experiment Status")
    print(f"{'='*60}")
    print(f"  Total      : {len(all_exps)}")
    print(f"  Done  ✓    : {len(done)}")
    print(f"  Error ✗    : {len(error)}")
    print(f"  Pending    : {len(todo)}")
    if error:
        print(f"\n  Failed experiments:")
        for e in error:
            print(f"    {e.experiment_id}")
    if todo:
        print(f"\n  Next pending: {todo[0].experiment_id}")


# ─────────────────────────────────────────────────────────────────────────────
# Run one experiment (used by SLURM array mode)
# ─────────────────────────────────────────────────────────────────────────────

def run_one(cfg: ExperimentConfig):
    """Run a single experiment, with error catching and markers."""
    from finetune import run_experiment

    exp_dir    = Path(gcfg.OUTPUT_DIR) / cfg.experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    err_marker = exp_dir / "ERROR"

    if is_done(cfg):
        print(f"  [SKIP] Already done: {cfg.experiment_id}")
        return True

    try:
        run_experiment(cfg, gcfg)
        return True
    except Exception as exc:
        import traceback
        err_log = {
            "experiment_id": cfg.experiment_id,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        with open(exp_dir / "error_log.json", "w") as f:
            json.dump(err_log, f, indent=2)
        err_marker.touch()
        print(f"  [ERROR] {cfg.experiment_id}: {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Run loop (sequential / local mode)
# ─────────────────────────────────────────────────────────────────────────────

def run_all(all_exps, start_idx=0, specific_ids=None):
    if specific_ids:
        id_set = set(specific_ids)
        to_run = [e for e in all_exps if e.experiment_id in id_set]
    else:
        to_run = all_exps[start_idx:]

    total  = len(to_run)
    n_done = n_error = 0
    t0     = time.time()

    print(f"\n{'='*60}")
    print(f"  Starting sequential sweep  ({total} experiments)")
    print(f"  Output dir : {gcfg.OUTPUT_DIR}")
    print(f"  Base model : {gcfg.BASE_MODEL_ID}")
    print(f"{'='*60}")

    for i, cfg in enumerate(to_run):
        print(f"\n  [{i+1}/{total}] {cfg.experiment_id}")
        ok = run_one(cfg)
        if ok:
            n_done += 1
        else:
            n_error += 1

        elapsed = time.time() - t0
        done_so_far = n_done + n_error
        if done_so_far > 0:
            eta = (elapsed / done_so_far) * (total - done_so_far)
            print(f"  Progress: {done_so_far}/{total}  ETA: {eta/60:.1f} min")

    print(f"\n  Sweep done — {n_done} OK, {n_error} errors  "
          f"({(time.time()-t0)/60:.1f} min total)")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment table
# ─────────────────────────────────────────────────────────────────────────────

def print_experiment_table(all_exps):
    print(f"\nAll {len(all_exps)} experiments:\n")
    print(f"{'#':>4}  {'Experiment ID':<70}  Done")
    print(f"{'─'*4}  {'─'*70}  {'─'*4}")
    for i, e in enumerate(all_exps):
        done_str = "✓" if is_done(e) else "·"
        print(f"{i:>4}  {e.experiment_id:<70}  {done_str}")


def export_ids(all_exps):
    """Print one experiment ID per line — useful for shell scripting."""
    for e in all_exps:
        print(e.experiment_id)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode   = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list",       action="store_true", help="List all experiment IDs with index")
    mode.add_argument("--export_ids", action="store_true", help="Print one ID per line (for scripts)")
    mode.add_argument("--run",        action="store_true", help="Run experiments")
    mode.add_argument("--status",     action="store_true", help="Show completion status")

    parser.add_argument("--start",      type=int,  default=0,  help="Start from index N (sequential mode)")
    parser.add_argument("--task_index", type=int,  default=None,
                        help="Run only experiment at this index (SLURM array mode)")
    parser.add_argument("--ids", nargs="+", default=None, help="Run specific experiment IDs")

    args     = parser.parse_args()
    all_exps = get_all_experiments()

    print(f"Total experiments in grid: {len(all_exps)}")

    if args.list:
        print_experiment_table(all_exps)

    elif args.export_ids:
        export_ids(all_exps)

    elif args.status:
        show_status(all_exps)

    elif args.run:
        # ── SLURM array mode: run exactly one experiment by index ──
        if args.task_index is not None:
            idx = args.task_index
            if idx < 0 or idx >= len(all_exps):
                print(f"ERROR: task_index {idx} out of range (0–{len(all_exps)-1})")
                sys.exit(1)
            cfg = all_exps[idx]
            print(f"SLURM array mode: running experiment [{idx}] {cfg.experiment_id}")
            ok  = run_one(cfg)
            sys.exit(0 if ok else 1)

        # ── Sequential / local mode ──
        else:
            run_all(all_exps, start_idx=args.start, specific_ids=args.ids)

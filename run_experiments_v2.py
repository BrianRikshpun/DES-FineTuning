"""
run_experiments.py  (v2 — multi-model support)
================================================
Orchestrates the full experiment sweep for any base model.

Usage:
    python3 run_experiments.py --run
    python3 run_experiments.py --run --config experiment_config_gemma.py
    python3 run_experiments.py --run --config experiment_config_qwen.py
    python3 run_experiments.py --run --task_index 0   # SLURM array mode
    python3 run_experiments.py --list                  # list all experiments
"""

import argparse
import importlib.util
import os
import sys
import time
import traceback
from pathlib import Path


def load_config(config_path=None):
    """Load experiment config from file path or default experiment_config.py."""
    if config_path is None:
        config_path = Path(__file__).parent / "experiment_config.py"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)

    spec   = importlib.util.spec_from_file_location("experiment_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def is_done(exp_dir):
    return (exp_dir / "DONE").exists()

def is_error(exp_dir):
    return (exp_dir / "ERROR").exists()

def mark_done(exp_dir):
    (exp_dir / "DONE").touch()

def mark_error(exp_dir, msg):
    (exp_dir / "ERROR").touch()
    with open(exp_dir / "error_log.txt", "w") as f:
        f.write(msg)


def run_single(cfg, global_cfg):
    """Run one experiment. Returns True on success, False on failure."""
    from finetune import run_finetune
    exp_dir = Path(global_cfg.OUTPUT_DIR) / cfg.experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    try:
        run_finetune(cfg, global_cfg, exp_dir)
        mark_done(exp_dir)
        return True
    except Exception as e:
        err_msg = traceback.format_exc()
        mark_error(exp_dir, err_msg)
        print(f"  [ERROR] {cfg.experiment_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",         action="store_true", help="Run all experiments")
    parser.add_argument("--list",        action="store_true", help="List all experiments")
    parser.add_argument("--task_index",  type=int, default=None, help="Run single experiment by index (SLURM array mode)")
    parser.add_argument("--config",      type=str, default=None, help="Path to experiment config file")
    args = parser.parse_args()

    global_cfg = load_config(args.config)
    experiments = global_cfg.get_all_experiments()

    if args.list:
        print(f"Total experiments: {len(experiments)}")
        for i, cfg in enumerate(experiments):
            done = is_done(Path(global_cfg.OUTPUT_DIR) / cfg.experiment_id)
            print(f"  [{i:4d}] {'✓' if done else ' '} {cfg.experiment_id}")
        return

    if args.task_index is not None:
        # SLURM array mode — run single experiment
        if args.task_index >= len(experiments):
            print(f"[ERROR] task_index {args.task_index} >= {len(experiments)}")
            sys.exit(1)
        cfg = experiments[args.task_index]
        exp_dir = Path(global_cfg.OUTPUT_DIR) / cfg.experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        print(f"SLURM array mode: running experiment [{args.task_index}] {cfg.experiment_id}")
        if is_done(exp_dir):
            print(f"  [SKIP] Already done: {cfg.experiment_id}")
            return
        success = run_single(cfg, global_cfg)
        sys.exit(0 if success else 1)

    if args.run:
        n_total   = len(experiments)
        n_done    = 0
        n_skipped = 0
        n_errors  = 0
        t_start   = time.time()

        print(f"Total experiments in grid: {n_total}")
        print(f"Output dir: {global_cfg.OUTPUT_DIR}")
        print(f"Base model: {global_cfg.BASE_MODEL_ID}\n")

        for i, cfg in enumerate(experiments):
            exp_dir = Path(global_cfg.OUTPUT_DIR) / cfg.experiment_id

            # Progress ETA
            elapsed  = time.time() - t_start
            done_so_far = n_done + n_skipped
            if done_so_far > 0:
                eta_s = (elapsed / done_so_far) * (n_total - done_so_far)
                eta_str = f"{eta_s/60:.1f} min"
            else:
                eta_str = "?"

            print(f"\n  [{i+1}/{n_total}] {cfg.experiment_id}")
            print(f"  Progress: {i}/{n_total}  ETA: {eta_str}")

            if is_done(exp_dir):
                print(f"  [SKIP] Already done: {cfg.experiment_id}")
                n_skipped += 1
                continue

            # Print config
            print(f"{'='*70}")
            print(f"  EXPERIMENT: {cfg.experiment_id}")
            print(f"{'='*70}")
            print(f"  dataset_size    : {cfg.dataset_size} per model")
            print(f"  quantization    : {cfg.quantization}")
            print(f"  lora_r          : {cfg.lora_r}")
            print(f"  lora_alpha      : {cfg.lora_alpha}")
            print(f"  lora_dropout    : {cfg.lora_dropout}")
            print(f"  target_modules  : {cfg.target_modules_label}")
            print(f"  learning_rate   : {cfg.learning_rate}")
            print(f"  num_epochs      : {cfg.num_epochs}")

            t_exp = time.time()
            success = run_single(cfg, global_cfg)

            if success:
                n_done += 1
                print(f"  ✓ Experiment done in {time.time()-t_exp:.0f}s  →  {exp_dir}")
            else:
                n_errors += 1

        print(f"\n{'='*70}")
        print(f"  Sweep done — {n_done} OK, {n_errors} errors  ({(time.time()-t_start)/60:.1f} min total)")
        print(f"{'='*70}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()

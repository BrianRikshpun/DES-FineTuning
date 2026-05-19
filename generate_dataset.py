"""
Dataset Generator for SimPy Simulation Surrogate Research
=========================================================
Generates 1000 Q&A pairs per simulation model with:
  - Randomized parameters sampled from defined ranges
  - Each config run N_SEEDS times with different seeds → answer = average output
  - 60/20/20 train/validation/test split
  - Output: dataset.json (all), dataset_train.json, dataset_val.json, dataset_test.json
"""

import importlib
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────
N_QUESTIONS_PER_MODEL = 1000
TRAIN_FRAC = 0.60
VAL_FRAC   = 0.20
TEST_FRAC  = 0.20
MASTER_SEED = 7
OUTPUT_DIR  = Path(__file__).parent

# Per-model seed counts (machine_shop is slower, use fewer seeds but still robust)
N_SEEDS_DEFAULT  = 10
N_SEEDS_OVERRIDE = {"machine_shop": 5}

SIM_MODULES = [
    "bank_renege",
    "carwash",
    "machine_shop",
    "gas_station",
    "movie_renege",
]

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_module(name):
    sim_dir = Path(__file__).parent / "simulations"
    sys.path.insert(0, str(sim_dir))
    mod = importlib.import_module(name)
    sys.path.pop(0)
    return mod


def sample_params(param_ranges, rng):
    params = {}
    for key, (lo, hi) in param_ranges.items():
        if isinstance(lo, int) and isinstance(hi, int):
            params[key] = rng.randint(lo, hi)
        else:
            params[key] = round(rng.uniform(lo, hi), 3)
    return params


def enforce_constraints(model_name, params):
    if model_name == "bank_renege":
        if params["max_patience"] < params["min_patience"]:
            params["max_patience"], params["min_patience"] = (
                params["min_patience"], params["max_patience"])
        if params["max_patience"] == params["min_patience"]:
            params["max_patience"] = params["min_patience"] + 0.5
    if model_name == "gas_station":
        if params["t_inter_max"] <= params["t_inter_min"]:
            params["t_inter_max"] = params["t_inter_min"] + 10
    return params


def average_runs(run_fn, params, n_seeds):
    accumulators = {}
    for i in range(n_seeds):
        result = run_fn(**params, seed=i * 13 + 37)
        for k, v in result.items():
            accumulators.setdefault(k, []).append(v)
    return {k: round(sum(v) / len(v), 4) for k, v in accumulators.items()}


def split_indices(n, train_f, val_f, rng):
    indices = list(range(n))
    rng.shuffle(indices)
    n_train = math.floor(n * train_f)
    n_val   = math.floor(n * val_f)
    return (
        set(indices[:n_train]),
        set(indices[n_train:n_train + n_val]),
        set(indices[n_train + n_val:]),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def generate():
    master_rng = random.Random(MASTER_SEED)
    all_records = []
    total_start = time.time()

    for model_name in SIM_MODULES:
        n_seeds = N_SEEDS_OVERRIDE.get(model_name, N_SEEDS_DEFAULT)
        print(f"\n{'='*60}")
        print(f"  Model: {model_name}  (n_seeds={n_seeds})")
        print(f"{'='*60}")

        mod          = load_module(model_name)
        param_ranges = mod.PARAM_RANGES
        output_key   = mod.OUTPUT_KEY
        question_tmpl = mod.QUESTION_TEMPLATE
        run_fn       = mod.run

        model_records = []
        n_errors = 0
        t0 = time.time()

        for idx in range(N_QUESTIONS_PER_MODEL):
            params = sample_params(param_ranges, master_rng)
            params = enforce_constraints(model_name, params)

            try:
                outputs = average_runs(run_fn, params, n_seeds)
            except Exception as e:
                n_errors += 1
                if n_errors <= 3:
                    print(f"  [WARN] idx={idx} error: {e}")
                continue

            answer   = outputs[output_key]
            question = question_tmpl.format(**params)

            model_records.append({
                "model": model_name,
                "question": question,
                "parameters": params,
                "outputs": outputs,
                "answer": answer,
                "output_key": output_key,
                "split": None,
            })

            if (idx + 1) % 200 == 0:
                elapsed = time.time() - t0
                rate = (idx + 1) / elapsed
                eta  = (N_QUESTIONS_PER_MODEL - idx - 1) / rate
                print(f"  [{model_name}] {idx+1:4d}/{N_QUESTIONS_PER_MODEL}  "
                      f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s  errors={n_errors}")

        elapsed = time.time() - t0
        print(f"  [{model_name}] Done: {len(model_records)} records in {elapsed:.1f}s  errors={n_errors}")

        # Assign splits
        n = len(model_records)
        train_idx, val_idx, test_idx = split_indices(n, TRAIN_FRAC, VAL_FRAC, master_rng)
        for i, rec in enumerate(model_records):
            rec["split"] = "train" if i in train_idx else ("validation" if i in val_idx else "test")

        all_records.extend(model_records)

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    full_path = OUTPUT_DIR / "dataset.json"
    with open(full_path, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"✓ Full dataset: {len(all_records)} records → {full_path}")

    for split_name in ("train", "validation", "test"):
        subset = [r for r in all_records if r["split"] == split_name]
        path   = OUTPUT_DIR / f"dataset_{split_name}.json"
        with open(path, "w") as f:
            json.dump(subset, f, indent=2)
        print(f"✓ {split_name:12s}: {len(subset):5d} records → {path}")

    print(f"\n── Per-model Summary ────────────────────────────────────────")
    for model_name in SIM_MODULES:
        subset   = [r for r in all_records if r["model"] == model_name]
        by_split = {s: sum(1 for r in subset if r["split"] == s)
                    for s in ("train", "validation", "test")}
        answers  = [r["answer"] for r in subset]
        print(f"  {model_name:<20} total={len(subset):4d}  "
              f"train={by_split['train']:3d}  val={by_split['validation']:3d}  "
              f"test={by_split['test']:3d}  "
              f"avg_answer={sum(answers)/len(answers):.3f}  "
              f"min={min(answers):.3f}  max={max(answers):.3f}")

    total_time = time.time() - total_start
    print(f"\nTotal generation time: {total_time:.1f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    generate()

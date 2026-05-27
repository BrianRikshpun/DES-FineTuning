"""
evaluate.py
===========
Step 3: Load each trained LoRA adapter, run inference on the test set,
and generate evaluation plots.

Outputs (all saved to OUTPUT_DIR/plots/):
  - predicted_vs_actual_<model>.png   per simulation model (best config)
  - r2_by_model.png                   R² bar chart across models
  - r2_by_dataset_size.png            dataset size effect
  - r2_by_quantization.png            4-bit vs 8-bit
  - r2_by_lora_rank.png               r=4,8,16,32
  - r2_by_target_modules.png          attention vs mlp vs both
  - r2_by_learning_rate.png           lr comparison
  - r2_heatmap_lr_vs_rank.png         heatmap
  - summary_table.csv                 full results table

Usage:
    python evaluate.py
    python evaluate.py --top_n 10        # only evaluate top 10 by val loss
    python evaluate.py --exp_id ds500_q4bit_r16_a32_do0.05_tmboth_lr2e-04_ep3
"""

import argparse
import json
import os
import re
import sys
import warnings
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")

# ── Paths (same as experiment_config.py) ─────────────────────────────────────
_USER       = os.environ.get("USER", "user")
EXPERIMENTS_DIR = Path(f"/home/{_USER}/llm_sim/experiments")
DATASET_DIR     = Path(f"/home/{_USER}/DES-FineTuning")
PLOTS_DIR       = EXPERIMENTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL_ID   = "meta-llama/Llama-3.2-1B"
MAX_SEQ_LENGTH  = 256
BATCH_SIZE      = 8     # inference batch size

SIM_MODELS = [
    "bank_renege",
    "carwash",
    "machine_shop",
    "gas_station",
    "movie_renege",
]

MODEL_COLORS = {
    "bank_renege":   "#378ADD",
    "carwash":       "#1D9E75",
    "machine_shop":  "#D85A30",
    "gas_station":   "#BA7517",
    "movie_renege":  "#7F77DD",
}

# ── Imports ───────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_test_data():
    path = DATASET_DIR / "dataset_test.json"
    with open(path) as f:
        records = json.load(f)
    return [r for r in records if r["model"] in SIM_MODELS]

def format_prompt(question):
    return f"### Simulation Question:\n{question}\n\n### Answer:\n"

def extract_number(text):
    """Extract first float from generated text."""
    matches = re.findall(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?", text)
    if matches:
        try:
            return float(matches[0])
        except ValueError:
            pass
    return None

def load_training_log(exp_dir):
    path = exp_dir / "training_log.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

def get_all_completed_experiments():
    """Return list of (exp_id, exp_dir, log) for all completed experiments."""
    results = []
    for d in sorted(EXPERIMENTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        if not (d / "DONE").exists():
            continue
        log = load_training_log(d)
        if log is None:
            continue
        results.append((d.name, d, log))
    return results

def get_final_val_loss(log):
    if log["eval_losses"]:
        return log["eval_losses"][-1]["eval_loss"]
    return float("inf")

# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(exp_dir, test_records):
    """
    Load adapter, run inference on test set, return list of
    {"model": ..., "true": ..., "pred": ...} dicts.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    adapter_path = exp_dir / "adapter"
    if not adapter_path.exists():
        print(f"  [WARN] No adapter at {adapter_path}, skipping.")
        return []

    log = load_training_log(exp_dir)
    quant = log["config"]["quantization"]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"   # for generation

    # Load base model
    if quant == "4bit":
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID, quantization_config=bnb_cfg, device_map="auto",
            trust_remote_code=True
        )
    elif quant == "8bit":
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID, quantization_config=bnb_cfg, device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID, torch_dtype=torch.float16, device_map="auto",
            trust_remote_code=True
        )

    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()

    results = []
    prompts  = [format_prompt(r["question"]) for r in test_records]
    trues    = [r["answer"] for r in test_records]
    sim_models = [r["model"] for r in test_records]

    # Batch inference
    for i in range(0, len(prompts), BATCH_SIZE):
        batch_prompts  = prompts[i:i+BATCH_SIZE]
        batch_trues    = trues[i:i+BATCH_SIZE]
        batch_models   = sim_models[i:i+BATCH_SIZE]

        enc = tokenizer(
            batch_prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=MAX_SEQ_LENGTH
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        for j, (output_ids, true_val, sim_name) in enumerate(
            zip(out, batch_trues, batch_models)
        ):
            input_len  = enc["input_ids"].shape[1]
            new_tokens = output_ids[input_len:]
            text       = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            pred       = extract_number(text)
            results.append({
                "model": sim_name,
                "true":  true_val,
                "pred":  pred,
                "text":  text,
            })

        if (i // BATCH_SIZE) % 10 == 0:
            print(f"    Inference: {min(i+BATCH_SIZE, len(prompts))}/{len(prompts)}")

    del model
    import gc
    gc.collect()
    if "torch" in sys.modules:
        import torch
        torch.cuda.empty_cache()

    return results

# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(results):
    """Compute R², MAE, RMSE overall and per simulation model."""
    valid = [r for r in results if r["pred"] is not None]
    if len(valid) < 2:
        return {"r2": None, "mae": None, "rmse": None, "n_valid": len(valid), "n_total": len(results), "by_model": {}}

    trues = np.array([r["true"] for r in valid])
    preds = np.array([r["pred"] for r in valid])

    overall = {
        "r2":      float(r2_score(trues, preds)),
        "mae":     float(mean_absolute_error(trues, preds)),
        "rmse":    float(np.sqrt(mean_squared_error(trues, preds))),
        "n_valid": len(valid),
        "n_total": len(results),
    }

    by_model = {}
    for sim in SIM_MODELS:
        sub = [r for r in valid if r["model"] == sim]
        if len(sub) < 2:
            by_model[sim] = {"r2": None, "mae": None, "rmse": None}
            continue
        t = np.array([r["true"] for r in sub])
        p = np.array([r["pred"] for r in sub])
        by_model[sim] = {
            "r2":   float(r2_score(t, p)),
            "mae":  float(mean_absolute_error(t, p)),
            "rmse": float(np.sqrt(mean_squared_error(t, p))),
            "n":    len(sub),
        }

    overall["by_model"] = by_model
    return overall

# ── Plots ─────────────────────────────────────────────────────────────────────

STYLE = dict(dpi=150, bbox_inches="tight")

def plot_predicted_vs_actual(results, exp_id, metrics):
    """One scatter plot per simulation model."""
    fig, axes = plt.subplots(1, len(SIM_MODELS), figsize=(18, 4))
    fig.suptitle(f"Predicted vs Actual — {exp_id}", fontsize=10, y=1.02)

    for ax, sim in zip(axes, SIM_MODELS):
        sub   = [r for r in results if r["model"] == sim and r["pred"] is not None]
        if not sub:
            ax.set_title(sim.replace("_", " "))
            continue
        trues = np.array([r["true"] for r in sub])
        preds = np.array([r["pred"] for r in sub])
        color = MODEL_COLORS[sim]
        r2    = metrics["by_model"].get(sim, {}).get("r2")

        ax.scatter(trues, preds, alpha=0.45, s=18, color=color, edgecolors="none")
        mn, mx = min(trues.min(), preds.min()), max(trues.max(), preds.max())
        ax.plot([mn, mx], [mn, mx], "k--", lw=1, alpha=0.5, label="perfect fit")
        ax.set_xlabel("Actual", fontsize=9)
        ax.set_ylabel("Predicted", fontsize=9)
        r2_str = f"R²={r2:.3f}" if r2 is not None else "R²=N/A"
        ax.set_title(f"{sim.replace('_',' ')}\n{r2_str}", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = PLOTS_DIR / f"predicted_vs_actual_{exp_id[:40]}.png"
    fig.savefig(path, **STYLE)
    plt.close(fig)
    return path

def plot_r2_by_model(all_results_df):
    """Bar chart: best R² per simulation model across all experiments."""
    best = all_results_df.groupby("sim_model")["r2"].max().reindex(SIM_MODELS)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors  = [MODEL_COLORS[m] for m in SIM_MODELS]
    bars    = ax.bar([m.replace("_", "\n") for m in SIM_MODELS], best.values,
                     color=colors, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, best.values):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("R² (best config per model)")
    ax.set_title("Best R² by Simulation Model")
    ax.axhline(0, color="black", lw=0.5)
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    path = PLOTS_DIR / "r2_by_model.png"
    fig.savefig(path, **STYLE)
    plt.close(fig)
    return path

def plot_r2_by_variable(all_results_df, col, title, xlabel, filename, order=None):
    """Generic: mean R² grouped by one config variable."""
    grp  = all_results_df.groupby(col)["r2"].agg(["mean", "std"]).reset_index()
    if order:
        grp = grp.set_index(col).reindex(order).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(grp[col].astype(str), grp["mean"],
           yerr=grp["std"], capsize=5,
           color="#378ADD", edgecolor="white", linewidth=0.5, error_kw=dict(lw=1))
    ax.set_ylim(0, max(1.0, (grp["mean"] + grp["std"]).max() + 0.1))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean R²")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    path = PLOTS_DIR / filename
    fig.savefig(path, **STYLE)
    plt.close(fig)
    return path

def plot_quantization_comparison(all_results_df):
    """Grouped bar: 4-bit vs 8-bit per simulation model."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x      = np.arange(len(SIM_MODELS))
    width  = 0.35
    for i, (quant, color) in enumerate([("4bit", "#378ADD"), ("8bit", "#D85A30")]):
        means = []
        for sim in SIM_MODELS:
            sub = all_results_df[(all_results_df["sim_model"] == sim) &
                                  (all_results_df["quantization"] == quant)]["r2"]
            means.append(sub.mean() if len(sub) > 0 else 0)
        ax.bar(x + i*width - width/2, means, width, label=quant, color=color,
               edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in SIM_MODELS], fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Mean R²")
    ax.set_title("4-bit vs 8-bit Quantization by Simulation Model")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    path = PLOTS_DIR / "r2_by_quantization.png"
    fig.savefig(path, **STYLE)
    plt.close(fig)
    return path

def plot_heatmap(all_results_df):
    """Heatmap: learning rate × LoRA rank → mean R²."""
    lrs   = sorted(all_results_df["learning_rate"].unique())
    ranks = sorted(all_results_df["lora_r"].unique())
    matrix = np.full((len(lrs), len(ranks)), np.nan)
    for i, lr in enumerate(lrs):
        for j, r in enumerate(ranks):
            sub = all_results_df[
                (all_results_df["learning_rate"] == lr) &
                (all_results_df["lora_r"] == r)
            ]["r2"]
            if len(sub) > 0:
                matrix[i, j] = sub.mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(ranks)))
    ax.set_xticklabels([f"r={r}" for r in ranks])
    ax.set_yticks(range(len(lrs)))
    ax.set_yticklabels([f"lr={lr:.0e}" for lr in lrs])
    ax.set_title("Mean R²: Learning Rate × LoRA Rank")
    plt.colorbar(im, ax=ax, label="Mean R²")
    for i in range(len(lrs)):
        for j in range(len(ranks)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=10, color="white" if val < 0.5 else "black")
    plt.tight_layout()
    path = PLOTS_DIR / "r2_heatmap_lr_vs_rank.png"
    fig.savefig(path, **STYLE)
    plt.close(fig)
    return path

def plot_target_modules(all_results_df):
    """Line plot: R² vs LoRA rank for each target module group."""
    ranks  = sorted(all_results_df["lora_r"].unique())
    colors = {"attention": "#378ADD", "mlp": "#1D9E75", "both": "#D85A30"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for tm, color in colors.items():
        means = []
        for r in ranks:
            sub = all_results_df[
                (all_results_df["target_modules"] == tm) &
                (all_results_df["lora_r"] == r)
            ]["r2"]
            means.append(sub.mean() if len(sub) > 0 else np.nan)
        ax.plot([f"r={r}" for r in ranks], means, marker="o", label=tm,
                color=color, linewidth=2, markersize=7)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("LoRA Rank")
    ax.set_ylabel("Mean R²")
    ax.set_title("R² vs LoRA Rank by Target Modules")
    ax.legend()
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    path = PLOTS_DIR / "r2_by_target_modules.png"
    fig.savefig(path, **STYLE)
    plt.close(fig)
    return path

def plot_learning_curves_grid(all_exps, top_n=9):
    """Grid of learning curves for top N experiments by final val loss."""
    ranked = sorted(all_exps, key=lambda x: get_final_val_loss(x[2]))[:top_n]
    ncols  = 3
    nrows  = (len(ranked) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.5))
    axes = axes.flatten()
    fig.suptitle(f"Learning Curves — Top {top_n} Experiments by Val Loss", fontsize=11)

    for ax, (exp_id, exp_dir, log) in zip(axes, ranked):
        train_steps  = [x["step"] for x in log["train_losses"]]
        train_losses = [x["loss"] for x in log["train_losses"]]
        eval_steps   = [x["step"] for x in log["eval_losses"]]
        eval_losses  = [x["eval_loss"] for x in log["eval_losses"]]
        ax.plot(train_steps, train_losses, color="#378ADD", lw=1.5, label="train")
        ax.plot(eval_steps, eval_losses, color="#D85A30", lw=2,
                ls="--", marker="o", ms=3, label="val")
        ax.set_title(exp_id.replace("_", " ")[:35], fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=6)

    for ax in axes[len(ranked):]:
        ax.set_visible(False)

    plt.tight_layout()
    path = PLOTS_DIR / "learning_curves_top.png"
    fig.savefig(path, **STYLE)
    plt.close(fig)
    return path

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id",   type=str, default=None,
                        help="Evaluate only this experiment ID")
    parser.add_argument("--top_n",    type=int, default=None,
                        help="Evaluate only top N experiments by val loss")
    parser.add_argument("--skip_inference", action="store_true",
                        help="Skip inference, only regenerate plots from saved results")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Step 3 — Evaluation")
    print(f"  Experiments dir : {EXPERIMENTS_DIR}")
    print(f"  Plots dir       : {PLOTS_DIR}")
    print(f"{'='*60}\n")

    all_exps = get_all_completed_experiments()
    print(f"Completed experiments found: {len(all_exps)}")

    # Filter if requested
    if args.exp_id:
        all_exps = [e for e in all_exps if e[0] == args.exp_id]
    elif args.top_n:
        all_exps = sorted(all_exps, key=lambda x: get_final_val_loss(x[2]))[:args.top_n]

    # Load test data
    test_records = load_test_data()
    print(f"Test records loaded: {len(test_records)}")

    # ── Run inference ─────────────────────────────────────────────────────────
    all_rows = []
    results_cache_path = EXPERIMENTS_DIR / "all_inference_results.json"

    if args.skip_inference and results_cache_path.exists():
        print("Loading cached inference results...")
        with open(results_cache_path) as f:
            all_rows = json.load(f)
    else:
        for i, (exp_id, exp_dir, log) in enumerate(all_exps):
            print(f"\n[{i+1}/{len(all_exps)}] Evaluating: {exp_id}")
            cfg = log["config"]

            results = run_inference(exp_dir, test_records)
            metrics = compute_metrics(results)

            # Save per-experiment predicted vs actual plot
            if results:
                plot_predicted_vs_actual(results, exp_id, metrics)

            # Save inference results
            inf_path = exp_dir / "inference_results.json"
            with open(inf_path, "w") as f:
                json.dump({"metrics": metrics, "results": results}, f)

            # Build summary row
            for sim in SIM_MODELS:
                m = metrics["by_model"].get(sim, {})
                all_rows.append({
                    "exp_id":         exp_id,
                    "sim_model":      sim,
                    "dataset_size":   cfg["dataset_size"],
                    "quantization":   cfg["quantization"],
                    "lora_r":         cfg["lora_r"],
                    "lora_alpha":     cfg["lora_alpha"],
                    "lora_dropout":   cfg["lora_dropout"],
                    "target_modules": cfg["target_modules"],
                    "learning_rate":  cfg["learning_rate"],
                    "num_epochs":     cfg["num_epochs"],
                    "r2":             m.get("r2"),
                    "mae":            m.get("mae"),
                    "rmse":           m.get("rmse"),
                    "val_loss":       get_final_val_loss(log),
                    "train_runtime_s": log.get("train_runtime_s"),
                })

            print(f"  Overall R²={metrics.get('r2', 'N/A')} | "
                  f"MAE={metrics.get('mae', 'N/A')} | "
                  f"n_valid={metrics.get('n_valid')}/{metrics.get('n_total')}")

        with open(results_cache_path, "w") as f:
            json.dump(all_rows, f)

    # ── Build summary DataFrame ───────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    df = df.dropna(subset=["r2"])

    csv_path = PLOTS_DIR / "summary_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSummary table saved → {csv_path}")

    # ── Generate summary plots ────────────────────────────────────────────────
    print("\nGenerating summary plots...")

    plot_r2_by_model(df)
    print("  ✓ r2_by_model.png")

    plot_r2_by_variable(df, "dataset_size", "Effect of Dataset Size on R²",
                         "Training samples per model",
                         "r2_by_dataset_size.png", order=[10, 100, 500, 1000, 1500])
    print("  ✓ r2_by_dataset_size.png")

    plot_r2_by_variable(df, "lora_r", "Effect of LoRA Rank on R²",
                         "LoRA rank (r)",
                         "r2_by_lora_rank.png", order=[4, 8, 16, 32])
    print("  ✓ r2_by_lora_rank.png")

    plot_r2_by_variable(df, "learning_rate", "Effect of Learning Rate on R²",
                         "Learning rate",
                         "r2_by_learning_rate.png")
    print("  ✓ r2_by_learning_rate.png")

    plot_quantization_comparison(df)
    print("  ✓ r2_by_quantization.png")

    plot_target_modules(df)
    print("  ✓ r2_by_target_modules.png")

    plot_heatmap(df)
    print("  ✓ r2_heatmap_lr_vs_rank.png")

    plot_learning_curves_grid(all_exps, top_n=9)
    print("  ✓ learning_curves_top.png")

    # ── Print top 10 experiments ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Top 10 experiments by mean R²")
    print(f"{'='*60}")
    top10 = (df.groupby("exp_id")["r2"]
               .mean()
               .sort_values(ascending=False)
               .head(10))
    for exp_id, r2 in top10.items():
        print(f"  {r2:.4f}  {exp_id}")

    print(f"\n{'='*60}")
    print(f"  All plots saved to: {PLOTS_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

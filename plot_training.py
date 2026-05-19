"""
plot_training.py
================
Reads training_log.json files produced by finetune.py and generates
learning-curve plots (train loss + validation loss vs. steps).

Usage:
    # Plot a single experiment
    python plot_training.py --exp_id ds10_q4bit_r8_a16_do0.05_tmattention_lr2e-04_ep3

    # Plot ALL completed experiments (one PNG per experiment)
    python plot_training.py --all

    # Overlay multiple experiments for comparison
    python plot_training.py --compare ds10_q4bit_r8 ds100_q4bit_r8 ds500_q4bit_r8
"""

import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
import experiment_config as gcfg

PLOTS_DIR = Path(gcfg.OUTPUT_DIR) / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Load log
# ─────────────────────────────────────────────────────────────────────────────

def load_log(exp_id: str) -> dict:
    path = Path(gcfg.OUTPUT_DIR) / exp_id / "training_log.json"
    if not path.exists():
        raise FileNotFoundError(f"No log found for {exp_id}: {path}")
    with open(path) as f:
        return json.load(f)


def load_all_logs() -> list:
    logs = []
    for d in sorted(Path(gcfg.OUTPUT_DIR).iterdir()):
        log_path = d / "training_log.json"
        if log_path.exists():
            with open(log_path) as f:
                logs.append(json.load(f))
    return logs


# ─────────────────────────────────────────────────────────────────────────────
# Single experiment plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_single(log: dict, save: bool = True, show: bool = False) -> Path:
    exp_id = log["experiment_id"]
    cfg    = log["config"]

    train_steps  = [x["step"] for x in log["train_losses"]]
    train_losses = [x["loss"] for x in log["train_losses"]]
    eval_steps   = [x["step"] for x in log["eval_losses"]]
    eval_losses  = [x["eval_loss"] for x in log["eval_losses"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training Curves\n{exp_id}", fontsize=10, y=1.01)

    # ── Left: raw loss curves ──────────────────────────────────────────────
    ax = axes[0]
    ax.plot(train_steps, train_losses, label="Train Loss",
            color="steelblue", linewidth=1.5, alpha=0.85)
    ax.plot(eval_steps, eval_losses, label="Val Loss",
            color="tomato", linewidth=2, linestyle="--", marker="o", markersize=4)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs. Steps")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Right: smoothed loss ───────────────────────────────────────────────
    ax2 = axes[1]

    def smooth(vals, w=5):
        if len(vals) < w:
            return vals
        kernel = np.ones(w) / w
        return np.convolve(vals, kernel, mode="valid").tolist()

    ax2.plot(train_steps[len(train_steps)-len(smooth(train_losses)):],
             smooth(train_losses), label="Train (smoothed)",
             color="steelblue", linewidth=2)
    ax2.plot(eval_steps, eval_losses, label="Val Loss",
             color="tomato", linewidth=2, linestyle="--", marker="o", markersize=4)
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Loss")
    ax2.set_title("Smoothed Loss vs. Steps")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Metadata text box
    meta = (
        f"dataset_size={cfg['dataset_size']} | quant={cfg['quantization']} | "
        f"r={cfg['lora_r']} | α={cfg['lora_alpha']}\n"
        f"dropout={cfg['lora_dropout']} | target={cfg['target_modules']} | "
        f"lr={cfg['learning_rate']:.0e} | epochs={cfg['num_epochs']}\n"
        f"train_samples={log['train_samples']} | "
        f"runtime={log['train_runtime_s']:.0f}s"
    )
    fig.text(0.5, -0.04, meta, ha="center", fontsize=8,
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()

    out_path = PLOTS_DIR / f"{exp_id}_learning_curves.png"
    if save:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {out_path}")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Comparison overlay plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_compare(exp_ids: list, variable_name: str = "comparison",
                 save: bool = True, show: bool = False) -> Path:
    """Overlay multiple experiments' val loss curves on one plot."""
    logs   = [load_log(e) for e in exp_ids]
    colors = cm.tab10(np.linspace(0, 0.9, len(logs)))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(f"Validation Loss Comparison ({variable_name})", fontsize=13)

    for log, color in zip(logs, colors):
        eval_steps  = [x["step"] for x in log["eval_losses"]]
        eval_losses = [x["eval_loss"] for x in log["eval_losses"]]
        label = log["experiment_id"]
        ax.plot(eval_steps, eval_losses, label=label, color=color,
                linewidth=2, marker="o", markersize=3)

    ax.set_xlabel("Steps")
    ax.set_ylabel("Validation Loss")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fname    = variable_name.replace(" ", "_")
    out_path = PLOTS_DIR / f"compare_{fname}.png"
    if save:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {out_path}")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Summary heatmap: final val loss across two variables
# ─────────────────────────────────────────────────────────────────────────────

def plot_heatmap_lr_vs_r(logs: list, quantization: str = "4bit",
                          dataset_size: int = 100, save: bool = True) -> Path:
    """
    Heatmap of final val loss for (learning_rate × lora_r)
    with quantization and dataset_size fixed.
    """
    import pandas as pd

    subset = [
        l for l in logs
        if l["config"]["quantization"] == quantization
        and l["config"]["dataset_size"] == dataset_size
        and l["eval_losses"]
    ]

    if not subset:
        print("  No data for heatmap with given filters.")
        return None

    rows = []
    for l in subset:
        final_val = l["eval_losses"][-1]["eval_loss"]
        rows.append({
            "lr": l["config"]["learning_rate"],
            "r":  l["config"]["lora_r"],
            "val_loss": final_val,
        })

    df   = pd.DataFrame(rows).groupby(["lr", "r"])["val_loss"].mean().reset_index()
    pivot = df.pivot(index="lr", columns="r", values="val_loss")

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"r={c}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"lr={r:.0e}" for r in pivot.index])
    ax.set_title(f"Final Val Loss: LR × LoRA-r  (quant={quantization}, ds={dataset_size})")
    plt.colorbar(im, ax=ax, label="Val Loss")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8,
                        color="white" if val > pivot.values.mean() else "black")

    plt.tight_layout()
    out_path = PLOTS_DIR / f"heatmap_lr_r_q{quantization}_ds{dataset_size}.png"
    if save:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {out_path}")
    plt.close(fig)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Summary bar chart: final val loss by dataset size
# ─────────────────────────────────────────────────────────────────────────────

def plot_dataset_size_effect(logs: list, save: bool = True) -> Path:
    from collections import defaultdict

    by_size = defaultdict(list)
    for l in logs:
        if l["eval_losses"]:
            final_val = l["eval_losses"][-1]["eval_loss"]
            by_size[l["config"]["dataset_size"]].append(final_val)

    sizes  = sorted(by_size.keys())
    means  = [np.mean(by_size[s]) for s in sizes]
    stds   = [np.std(by_size[s])  for s in sizes]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([str(s) for s in sizes], means, yerr=stds,
           color=["#4c9be8", "#f08030", "#3cb371"],
           capsize=6, edgecolor="black", linewidth=0.7)
    ax.set_xlabel("Training Samples per Simulation Model")
    ax.set_ylabel("Final Validation Loss (mean ± std)")
    ax.set_title("Effect of Dataset Size on Validation Loss")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    out_path = PLOTS_DIR / "dataset_size_effect.png"
    if save:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {out_path}")
    plt.close(fig)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--exp_id",  type=str, help="Single experiment ID")
    group.add_argument("--all",     action="store_true", help="Plot all completed experiments")
    group.add_argument("--compare", nargs="+", help="Overlay multiple experiment IDs")
    group.add_argument("--summary", action="store_true", help="Generate summary plots")
    args = parser.parse_args()

    if args.exp_id:
        log = load_log(args.exp_id)
        plot_single(log, save=True)

    elif args.all:
        logs = load_all_logs()
        print(f"Plotting {len(logs)} experiments...")
        for log in logs:
            try:
                plot_single(log, save=True)
            except Exception as e:
                print(f"  [WARN] {log['experiment_id']}: {e}")

    elif args.compare:
        plot_compare(args.compare, variable_name="_vs_".join(args.compare[:3]))

    elif args.summary:
        logs = load_all_logs()
        print(f"Generating summary plots from {len(logs)} experiments...")
        plot_dataset_size_effect(logs)
        for quant in gcfg.QUANTIZATIONS:
            for ds in gcfg.DATASET_SIZES:
                plot_heatmap_lr_vs_r(logs, quantization=quant, dataset_size=ds)

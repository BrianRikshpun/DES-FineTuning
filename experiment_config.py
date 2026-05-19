"""
experiment_config.py
====================
Central configuration for ALL fine-tuning experiments.

Edit this file to add/remove independent variables before running.
Every combination of the lists below will become one experiment run.
The cartesian product is computed in run_experiments.py.
"""

from itertools import product
from dataclasses import dataclass, field
from typing import List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# PATHS  (edit these to match your environment)
# ─────────────────────────────────────────────────────────────────────────────
DATASET_DIR   = "../step1_generate_dataset"   # folder with dataset_*.json files
OUTPUT_DIR    = "./experiments"               # where checkpoints + logs land
HF_TOKEN      = None                          # set to your HuggingFace token string
                                              # or leave None if already logged in via `huggingface-cli login`

# ─────────────────────────────────────────────────────────────────────────────
# BASE MODEL
# ─────────────────────────────────────────────────────────────────────────────
# Only LLaMA-family models are guaranteed to work with this pipeline.
# Swap in any HF model ID (e.g. "meta-llama/Llama-3.2-3B").
BASE_MODEL_ID = "meta-llama/Llama-3.2-3B"

# ─────────────────────────────────────────────────────────────────────────────
# INDEPENDENT VARIABLES  (each list = the levels to sweep for that variable)
# Comment out any line (or set its list to a single value) to hold it constant.
# ─────────────────────────────────────────────────────────────────────────────

# ── Dataset size (questions per simulation model kept in training set) ────────
# The generator produced 600 train records per model (3000 total).
# We sub-sample to these sizes for the sweep.
DATASET_SIZES = [10, 100, 500]           # questions PER MODEL in training

# ── Quantization ─────────────────────────────────────────────────────────────
# "4bit"  → bitsandbytes NF4 QLoRA
# "8bit"  → bitsandbytes int8
# "none"  → fp16 full (requires more VRAM; skip if GPU < 24 GB)
QUANTIZATIONS = ["4bit", "8bit"]         # add "none" if you have ≥24 GB VRAM

# ── LoRA rank r ──────────────────────────────────────────────────────────────
LORA_RANKS = [4, 8, 16, 32]

# ── LoRA alpha (scaling factor; common choices: r, 2r) ───────────────────────
# Use "auto_2r" to always set alpha = 2 * r at runtime.
LORA_ALPHAS = ["auto_2r"]                # or e.g. [16, 32, 64]

# ── LoRA dropout ─────────────────────────────────────────────────────────────
LORA_DROPOUTS = [0.05, 0.1]

# ── Target modules (which layers in the transformer to adapt) ─────────────────
# "attention"   → q_proj, k_proj, v_proj, o_proj
# "mlp"         → gate_proj, up_proj, down_proj
# "both"        → all of the above
TARGET_MODULES_OPTIONS = ["attention", "mlp", "both"]

# ── Learning rate ─────────────────────────────────────────────────────────────
LEARNING_RATES = [1e-4, 2e-4, 5e-4]

# ── Training epochs ───────────────────────────────────────────────────────────
NUM_EPOCHS = [3]                         # keep fixed unless you want to sweep it

# ── Batch size (per device) ───────────────────────────────────────────────────
PER_DEVICE_BATCH_SIZE = 4               # lower if OOM (try 2 or 1)
GRADIENT_ACCUMULATION_STEPS = 4        # effective batch = batch_size * accum

# ─────────────────────────────────────────────────────────────────────────────
# FIXED TRAINING HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
MAX_SEQ_LENGTH   = 256      # max token length for input+output
WARMUP_RATIO     = 0.05
LR_SCHEDULER     = "cosine"
WEIGHT_DECAY     = 0.01
LOGGING_STEPS    = 10
SAVE_STEPS       = 100
EVAL_STEPS       = 50
FP16             = True     # set False if your GPU doesn't support fp16
BF16             = False    # set True for Ampere+ GPUs (A100, 3090, 4090…)

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION MODELS TO INCLUDE  (subset of what was generated)
# ─────────────────────────────────────────────────────────────────────────────
# All 5 models will be in the training data mixed together.
# You can filter to a subset here.
SIM_MODELS = [
    "bank_renege",
    "carwash",
    "machine_shop",
    "gas_station",
    "movie_renege",
]

# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT GRID  (auto-computed — do not edit)
# ─────────────────────────────────────────────────────────────────────────────
TARGET_MODULE_MAP = {
    "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mlp":       ["gate_proj", "up_proj", "down_proj"],
    "both":      ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"],
}


@dataclass
class ExperimentConfig:
    dataset_size:     int
    quantization:     str
    lora_r:           int
    lora_alpha:       int       # resolved (never "auto_2r")
    lora_dropout:     float
    target_modules:   List[str]
    target_modules_label: str   # "attention" / "mlp" / "both"
    learning_rate:    float
    num_epochs:       int
    experiment_id:    str = ""  # filled in by get_all_experiments()

    def short_id(self):
        return (
            f"ds{self.dataset_size}"
            f"_q{self.quantization}"
            f"_r{self.lora_r}"
            f"_a{self.lora_alpha}"
            f"_do{self.lora_dropout}"
            f"_tm{self.target_modules_label}"
            f"_lr{self.learning_rate:.0e}"
            f"_ep{self.num_epochs}"
        )


def get_all_experiments() -> List[ExperimentConfig]:
    """Return the full cartesian product of all sweep variables."""
    configs = []
    combo_vars = [
        DATASET_SIZES,
        QUANTIZATIONS,
        LORA_RANKS,
        LORA_ALPHAS,
        LORA_DROPOUTS,
        TARGET_MODULES_OPTIONS,
        LEARNING_RATES,
        NUM_EPOCHS,
    ]
    for ds, quant, r, alpha_spec, dropout, tm_label, lr, epochs in product(*combo_vars):
        alpha = 2 * r if alpha_spec == "auto_2r" else int(alpha_spec)
        cfg = ExperimentConfig(
            dataset_size         = ds,
            quantization         = quant,
            lora_r               = r,
            lora_alpha           = alpha,
            lora_dropout         = dropout,
            target_modules       = TARGET_MODULE_MAP[tm_label],
            target_modules_label = tm_label,
            learning_rate        = lr,
            num_epochs           = epochs,
        )
        cfg.experiment_id = cfg.short_id()
        configs.append(cfg)
    return configs


if __name__ == "__main__":
    exps = get_all_experiments()
    print(f"Total experiments in grid: {len(exps)}")
    print("\nFirst 5 experiment IDs:")
    for e in exps[:5]:
        print(f"  {e.experiment_id}")
    print("\nLast 5 experiment IDs:")
    for e in exps[-5:]:
        print(f"  {e.experiment_id}")

    # Break down by variable
    print(f"\n{'─'*50}")
    print(f"  dataset_sizes        : {DATASET_SIZES}")
    print(f"  quantizations        : {QUANTIZATIONS}")
    print(f"  lora_ranks           : {LORA_RANKS}")
    print(f"  lora_alphas          : {LORA_ALPHAS}")
    print(f"  lora_dropouts        : {LORA_DROPOUTS}")
    print(f"  target_modules       : {TARGET_MODULES_OPTIONS}")
    print(f"  learning_rates       : {LEARNING_RATES}")
    print(f"  num_epochs           : {NUM_EPOCHS}")

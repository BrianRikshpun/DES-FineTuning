"""
experiment_config_hamming.py
============================
Drop-in replacement for experiment_config.py tuned for the NPS Hamming HPC cluster.

On Hamming, copy this file over experiment_config.py:
    cp experiment_config_hamming.py experiment_config.py

Key differences from the local version:
  - Paths use /scratch/$USER (fast parallel filesystem on Hamming)
  - BF16=True  (Hamming GPUs are A100/H100-class → bfloat16 is preferred)
  - FP16=False (don't mix both)
  - HF model cache redirected to /scratch so it survives between jobs
"""

import os
from itertools import product
from dataclasses import dataclass, field
from typing import List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# PATHS  (Hamming-specific)
# ─────────────────────────────────────────────────────────────────────────────
_USER        = os.environ.get("USER", "user")
_SCRATCH     = f"/scratch/{_USER}/llm_sim"

DATASET_DIR  = f"{_SCRATCH}/repo/step1_generate_dataset"
OUTPUT_DIR   = f"{_SCRATCH}/experiments"
HF_TOKEN     = None   # set to your token string if model is gated

# Tell HuggingFace to cache model weights on scratch (not home quota)
os.environ["HF_HOME"]            = f"{_SCRATCH}/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = f"{_SCRATCH}/hf_cache"
os.environ["HF_DATASETS_CACHE"]  = f"{_SCRATCH}/hf_cache"

# ─────────────────────────────────────────────────────────────────────────────
# BASE MODEL
# ─────────────────────────────────────────────────────────────────────────────
BASE_MODEL_ID = "meta-llama/Llama-3.2-3B"

# ─────────────────────────────────────────────────────────────────────────────
# SWEEP VARIABLES
# ─────────────────────────────────────────────────────────────────────────────
DATASET_SIZES          = [10, 100, 500]
QUANTIZATIONS          = ["4bit", "8bit"]
LORA_RANKS             = [4, 8, 16, 32]
LORA_ALPHAS            = ["auto_2r"]
LORA_DROPOUTS          = [0.05, 0.1]
TARGET_MODULES_OPTIONS = ["attention", "mlp", "both"]
LEARNING_RATES         = [1e-4, 2e-4, 5e-4]
NUM_EPOCHS             = [3]

# ─────────────────────────────────────────────────────────────────────────────
# FIXED HYPERPARAMETERS  (tuned for Hamming A100/H100 GPUs)
# ─────────────────────────────────────────────────────────────────────────────
PER_DEVICE_BATCH_SIZE       = 8     # A100 80 GB → larger batch than local
GRADIENT_ACCUMULATION_STEPS = 2     # effective batch = 16
MAX_SEQ_LENGTH              = 256
WARMUP_RATIO                = 0.05
LR_SCHEDULER                = "cosine"
WEIGHT_DECAY                = 0.01
LOGGING_STEPS               = 10
SAVE_STEPS                  = 100
EVAL_STEPS                  = 50
FP16                        = False  # don't use on A100/H100
BF16                        = True   # A100/H100 natively support bfloat16

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION MODELS
# ─────────────────────────────────────────────────────────────────────────────
SIM_MODELS = [
    "bank_renege",
    "carwash",
    "machine_shop",
    "gas_station",
    "movie_renege",
]

# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT GRID  (same logic as local config)
# ─────────────────────────────────────────────────────────────────────────────
TARGET_MODULE_MAP = {
    "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mlp":       ["gate_proj", "up_proj", "down_proj"],
    "both":      ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"],
}


@dataclass
class ExperimentConfig:
    dataset_size:         int
    quantization:         str
    lora_r:               int
    lora_alpha:           int
    lora_dropout:         float
    target_modules:       List[str]
    target_modules_label: str
    learning_rate:        float
    num_epochs:           int
    experiment_id:        str = ""

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
    configs = []
    for ds, quant, r, alpha_spec, dropout, tm_label, lr, epochs in product(
        DATASET_SIZES, QUANTIZATIONS, LORA_RANKS, LORA_ALPHAS,
        LORA_DROPOUTS, TARGET_MODULES_OPTIONS, LEARNING_RATES, NUM_EPOCHS
    ):
        alpha = 2 * r if alpha_spec == "auto_2r" else int(alpha_spec)
        cfg   = ExperimentConfig(
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
    print(f"Total experiments : {len(exps)}")
    print(f"Output dir        : {OUTPUT_DIR}")
    print(f"Dataset dir       : {DATASET_DIR}")
    print(f"HF cache          : {os.environ['HF_HOME']}")

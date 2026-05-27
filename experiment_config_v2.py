"""
experiment_config.py — v2
=========================
Adds dataset sizes 1000 and 1500 per simulation model
to the original sweep of [10, 100, 500].

New experiments:
  2 new dataset sizes (1000, 1500)
  × 2 quantizations
  × 4 ranks
  × 1 alpha
  × 2 dropouts
  × 3 target modules
  × 3 learning rates
  × 1 epoch setting
  = 288 new experiments

Combined with original 432 = 720 total
"""

import os
from itertools import product
from dataclasses import dataclass
from typing import List

_USER    = os.environ.get("USER", "user")
_SCRATCH = f"/home/{_USER}"

DATASET_DIR = f"{_SCRATCH}/DES-FineTuning"
OUTPUT_DIR  = f"{_SCRATCH}/llm_sim/experiments"
HF_TOKEN    = None

os.environ["HF_HOME"]            = f"{_SCRATCH}/llm_sim/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = f"{_SCRATCH}/llm_sim/hf_cache"
os.environ["HF_DATASETS_CACHE"]  = f"{_SCRATCH}/llm_sim/hf_cache"

BASE_MODEL_ID = "meta-llama/Llama-3.2-1B"

# ── Sweep variables ────────────────────────────────────────────────────────────
# Original: [10, 100, 500]  +  New: [1000, 1500]
DATASET_SIZES          = [10, 100, 500, 1000, 1500]

QUANTIZATIONS          = ["4bit", "8bit"]
LORA_RANKS             = [4, 8, 16, 32]
LORA_ALPHAS            = ["auto_2r"]
LORA_DROPOUTS          = [0.05, 0.1]
TARGET_MODULES_OPTIONS = ["attention", "mlp", "both"]
LEARNING_RATES         = [1e-4, 2e-4, 5e-4]
NUM_EPOCHS             = [3]

# ── Fixed hyperparameters ──────────────────────────────────────────────────────
PER_DEVICE_BATCH_SIZE       = 4
GRADIENT_ACCUMULATION_STEPS = 4
MAX_SEQ_LENGTH              = 256
WARMUP_RATIO                = 0.05
LR_SCHEDULER                = "cosine"
WEIGHT_DECAY                = 0.01
LOGGING_STEPS               = 10
SAVE_STEPS                  = 200
EVAL_STEPS                  = 100
FP16                        = True
BF16                        = False

SIM_MODELS = [
    "bank_renege",
    "carwash",
    "machine_shop",
    "gas_station",
    "movie_renege",
]

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
    exps     = get_all_experiments()
    original = [e for e in exps if e.dataset_size in [10, 100, 500]]
    new_runs = [e for e in exps if e.dataset_size in [1000, 1500]]
    ds1000   = [e for e in new_runs if e.dataset_size == 1000]
    ds1500   = [e for e in new_runs if e.dataset_size == 1500]

    print(f"Original experiments (ds10/100/500) : {len(original)}")
    print(f"New ds1000 experiments               : {len(ds1000)}")
    print(f"New ds1500 experiments               : {len(ds1500)}")
    print(f"Total experiments                    : {len(exps)}")
    print(f"\nSample new experiment IDs:")
    for e in new_runs[:5]:
        print(f"  {e.experiment_id}")
    print(f"\nDataset sizes per model → total training examples:")
    for ds in [10, 100, 500, 1000, 1500]:
        print(f"  ds{ds:4d} per model → {ds*5:5d} total training examples")
    print(f"\nNote: original 432 DONE markers preserved — only 288 new runs execute.")

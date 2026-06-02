"""
finetune.py  (v3 — multi-model support)
========================================
Handles LLaMA, Gemma 3, and Qwen 2.5 with model-specific fixes:

  LLaMA  → bias="none", standard tokenizer setup
  Gemma  → bias="none", requires trust_remote_code, pad_token fix
  Qwen   → bias="all"  (QKV bias present), pad_token fix
"""

import gc
import json
import os
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


# ── Prompt format ─────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = "### Simulation Question:\n{question}\n\n### Answer:\n{answer}"
RESPONSE_TEMPLATE = "### Answer:\n"


def format_prompt(question, answer=None):
    if answer is None:
        return f"### Simulation Question:\n{question}\n\n### Answer:\n"
    return PROMPT_TEMPLATE.format(question=question, answer=str(round(float(answer), 4)))


# ── Model-specific helpers ─────────────────────────────────────────────────────

def get_model_family(base_model_id):
    """Detect model family from model ID string."""
    mid = base_model_id.lower()
    if "llama" in mid:
        return "llama"
    if "gemma" in mid:
        return "gemma"
    if "qwen" in mid:
        return "qwen"
    return "unknown"


def get_lora_bias(model_family):
    """
    Qwen uses QKV bias — set bias='all' to also adapt bias terms.
    LLaMA and Gemma have no QKV bias — use 'none'.
    """
    return "all" if model_family == "qwen" else "none"


def setup_tokenizer(tokenizer, model_family):
    """
    Apply model-specific tokenizer fixes.
    - All models: ensure pad token is set
    - Gemma: uses <eos> as pad (same as LLaMA)
    - Qwen: has its own pad token but we normalize to eos
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Left-padding for generation (right-padding for training)
    tokenizer.padding_side = "right"

    return tokenizer


def load_base_model(base_model_id, quantization, model_family):
    """Load quantized base model with model-specific settings."""
    common_kwargs = {
        "device_map":        "auto",
        "trust_remote_code": True,   # needed for Gemma and Qwen
    }

    if quantization == "4bit":
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id, quantization_config=bnb_cfg, **common_kwargs
        )
    elif quantization == "8bit":
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id, quantization_config=bnb_cfg, **common_kwargs
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id, torch_dtype=torch.float16, **common_kwargs
        )

    return model


# ── Loss tracking callback ─────────────────────────────────────────────────────

class LossCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []
        self.eval_losses  = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        if "loss" in logs:
            self.train_losses.append({"step": step, "loss": logs["loss"]})
        if "eval_loss" in logs:
            self.eval_losses.append({"step": step, "eval_loss": logs["eval_loss"]})


# ── Dataset loading ────────────────────────────────────────────────────────────

def load_split(dataset_dir, split, dataset_size, sim_models, seed=42):
    """Load and subsample a dataset split."""
    import json
    import random

    path = Path(dataset_dir) / f"dataset_{split}.json"
    with open(path) as f:
        records = json.load(f)

    records = [r for r in records if r["model"] in sim_models]

    if split == "train" and dataset_size is not None:
        rng = random.Random(seed)
        result = []
        for model in sim_models:
            subset = [r for r in records if r["model"] == model]
            rng.shuffle(subset)
            result.extend(subset[:dataset_size])
        rng.shuffle(result)
        records = result
    elif split == "validation" and dataset_size is not None:
        # Scale validation to 20% of train size, min 10 per model
        val_per_model = max(10, dataset_size // 5)
        rng = random.Random(seed + 1)
        result = []
        for model in sim_models:
            subset = [r for r in records if r["model"] == model]
            rng.shuffle(subset)
            result.extend(subset[:val_per_model])
        records = result

    return records


def records_to_dataset(records, tokenizer, max_seq_length):
    """Convert records to HuggingFace Dataset with formatted prompts."""
    texts = [
        format_prompt(r["question"], r["answer"])
        for r in records
    ]
    return Dataset.from_dict({"text": texts})


# ── Main training function ─────────────────────────────────────────────────────

def run_finetune(cfg, global_cfg, exp_dir):
    """
    Full QLoRA fine-tuning for one experiment configuration.
    Handles LLaMA, Gemma 3, and Qwen 2.5 automatically.
    """
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    model_family = getattr(global_cfg, "MODEL_FAMILY", None) or get_model_family(global_cfg.BASE_MODEL_ID)
    lora_bias    = get_lora_bias(model_family)

    print(f"  Model family    : {model_family}")
    print(f"  LoRA bias       : {lora_bias}")

    # ── Load tokenizer ────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        global_cfg.BASE_MODEL_ID,
        trust_remote_code=True,
        use_fast=True,
    )
    tokenizer = setup_tokenizer(tokenizer, model_family)

    # ── Load datasets ─────────────────────────────────────────────────────────
    train_records = load_split(global_cfg.DATASET_DIR, "train",
                               cfg.dataset_size, global_cfg.SIM_MODELS)
    val_records   = load_split(global_cfg.DATASET_DIR, "validation",
                               cfg.dataset_size, global_cfg.SIM_MODELS)
    test_records  = load_split(global_cfg.DATASET_DIR, "test",
                               None, global_cfg.SIM_MODELS)

    print(f"  Dataset sizes  → train={len(train_records)}  "
          f"val={len(val_records)}  test={len(test_records)}")

    train_ds = records_to_dataset(train_records, tokenizer, global_cfg.MAX_SEQ_LENGTH)
    val_ds   = records_to_dataset(val_records,   tokenizer, global_cfg.MAX_SEQ_LENGTH)

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_base_model(global_cfg.BASE_MODEL_ID, cfg.quantization, model_family)
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    # ── LoRA config ───────────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        r               = cfg.lora_r,
        lora_alpha      = cfg.lora_alpha,
        lora_dropout    = cfg.lora_dropout,
        target_modules  = cfg.target_modules,
        bias            = lora_bias,
        task_type       = TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Data collator (loss masking) ──────────────────────────────────────────
    # Mask question tokens, compute loss only on answer tokens
    response_token_ids = tokenizer.encode(
        RESPONSE_TEMPLATE, add_special_tokens=False
    )
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_token_ids,
        tokenizer=tokenizer,
    )

    # ── Training arguments ────────────────────────────────────────────────────
    loss_cb = LossCallback()

    checkpoint_dir = exp_dir / "checkpoints"
    training_args  = TrainingArguments(
        output_dir                  = str(checkpoint_dir),
        num_train_epochs            = cfg.num_epochs,
        per_device_train_batch_size = global_cfg.PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size  = global_cfg.PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps = global_cfg.GRADIENT_ACCUMULATION_STEPS,
        learning_rate               = cfg.learning_rate,
        lr_scheduler_type           = global_cfg.LR_SCHEDULER,
        warmup_ratio                = global_cfg.WARMUP_RATIO,
        weight_decay                = global_cfg.WEIGHT_DECAY,
        fp16                        = global_cfg.FP16,
        bf16                        = global_cfg.BF16,
        logging_steps               = global_cfg.LOGGING_STEPS,
        eval_strategy               = "steps",
        eval_steps                  = global_cfg.EVAL_STEPS,
        save_strategy               = "steps",
        save_steps                  = global_cfg.SAVE_STEPS,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        report_to                   = "none",
        dataloader_pin_memory       = False,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        data_collator   = collator,
        dataset_text_field = "text",
        max_seq_length  = global_cfg.MAX_SEQ_LENGTH,
        callbacks       = [loss_cb],
    )

    t_start = time.time()
    trainer.train()
    runtime = time.time() - t_start

    # ── Save adapter ──────────────────────────────────────────────────────────
    adapter_dir = exp_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    # ── Save training log ─────────────────────────────────────────────────────
    log = {
        "experiment_id": cfg.experiment_id,
        "base_model_id": global_cfg.BASE_MODEL_ID,
        "model_family":  model_family,
        "config": {
            "dataset_size":    cfg.dataset_size,
            "quantization":    cfg.quantization,
            "lora_r":          cfg.lora_r,
            "lora_alpha":      cfg.lora_alpha,
            "lora_dropout":    cfg.lora_dropout,
            "target_modules":  cfg.target_modules_label,
            "learning_rate":   cfg.learning_rate,
            "num_epochs":      cfg.num_epochs,
            "lora_bias":       lora_bias,
        },
        "train_losses":    loss_cb.train_losses,
        "eval_losses":     loss_cb.eval_losses,
        "train_runtime_s": round(runtime, 1),
        "n_train":         len(train_records),
        "n_val":           len(val_records),
    }
    with open(exp_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

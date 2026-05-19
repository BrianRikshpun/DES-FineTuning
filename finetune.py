"""
finetune.py
===========
Core fine-tuning engine.  Called once per experiment configuration.

Usage (standalone):
    python finetune.py --experiment_id ds10_q4bit_r8_a16_do0.05_tmattention_lr2e-04_ep3

Usage (from run_experiments.py):
    from finetune import run_experiment
    run_experiment(cfg)
"""

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Lazy imports — only pulled in when actually needed
# ─────────────────────────────────────────────────────────────────────────────

def _import_training_deps():
    """Import heavy ML deps once at training time."""
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForSeq2Seq,
        BitsAndBytesConfig,
        EarlyStoppingCallback,
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        TaskType,
    )
    return (
        AutoModelForCausalLM, AutoTokenizer,
        TrainingArguments, Trainer,
        DataCollatorForSeq2Seq, BitsAndBytesConfig,
        EarlyStoppingCallback,
        LoraConfig, get_peft_model,
        prepare_model_for_kbit_training, TaskType,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(cfg, base_model_id: str, hf_token=None):
    (
        AutoModelForCausalLM, AutoTokenizer,
        TrainingArguments, Trainer,
        DataCollatorForSeq2Seq, BitsAndBytesConfig,
        EarlyStoppingCallback,
        LoraConfig, get_peft_model,
        prepare_model_for_kbit_training, TaskType,
    ) = _import_training_deps()

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        token=hf_token,
        trust_remote_code=True,
    )
    tokenizer.pad_token     = tokenizer.eos_token
    tokenizer.padding_side  = "right"

    # ── Quantization config ───────────────────────────────────────────────────
    if cfg.quantization == "4bit":
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit               = True,
            bnb_4bit_use_double_quant  = True,
            bnb_4bit_quant_type        = "nf4",
            bnb_4bit_compute_dtype     = torch.float16,
        )
        model_kwargs = dict(quantization_config=bnb_cfg, device_map="auto")
    elif cfg.quantization == "8bit":
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
        model_kwargs = dict(quantization_config=bnb_cfg, device_map="auto")
    else:  # "none" → fp16
        model_kwargs = dict(torch_dtype=torch.float16, device_map="auto")

    # ── Base model ────────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        token=hf_token,
        trust_remote_code=True,
        **model_kwargs,
    )

    # Prepare for k-bit training (adds gradient checkpointing, etc.)
    if cfg.quantization in ("4bit", "8bit"):
        model = prepare_model_for_kbit_training(model)

    model.config.use_cache = False   # required for gradient checkpointing

    # ── LoRA ──────────────────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        r                = cfg.lora_r,
        lora_alpha       = cfg.lora_alpha,
        lora_dropout     = cfg.lora_dropout,
        target_modules   = cfg.target_modules,
        bias             = "none",
        task_type        = TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Custom loss-tracking callback
# ─────────────────────────────────────────────────────────────────────────────

class LossHistoryCallback:
    """Collects train/eval loss at every logging step."""

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


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(cfg, global_cfg=None):
    """
    Train one QLoRA experiment defined by `cfg` (ExperimentConfig).
    `global_cfg` can be the experiment_config module for paths/hyperparams.
    """
    if global_cfg is None:
        import experiment_config as global_cfg

    exp_dir = Path(global_cfg.OUTPUT_DIR) / cfg.experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    log_path    = exp_dir / "training_log.json"
    done_marker = exp_dir / "DONE"

    # Skip if already finished
    if done_marker.exists():
        print(f"  [SKIP] {cfg.experiment_id} already done.")
        return

    print(f"\n{'='*70}")
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

    t_start = time.time()

    # ── Load model & tokenizer ────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(
        cfg, global_cfg.BASE_MODEL_ID, global_cfg.HF_TOKEN
    )

    # ── Build datasets ────────────────────────────────────────────────────────
    from sim_dataset import build_datasets
    train_ds, val_ds, test_ds, raw_test = build_datasets(
        dataset_dir            = global_cfg.DATASET_DIR,
        tokenizer              = tokenizer,
        dataset_size_per_model = cfg.dataset_size,
        sim_models             = global_cfg.SIM_MODELS,
        max_length             = global_cfg.MAX_SEQ_LENGTH,
    )

    # ── Data collator ─────────────────────────────────────────────────────────
    from transformers import DataCollatorForSeq2Seq
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    # ── Training arguments ────────────────────────────────────────────────────
    from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

    training_args = TrainingArguments(
        output_dir                  = str(exp_dir / "checkpoints"),
        num_train_epochs            = cfg.num_epochs,
        per_device_train_batch_size = global_cfg.PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size  = global_cfg.PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps = global_cfg.GRADIENT_ACCUMULATION_STEPS,
        learning_rate               = cfg.learning_rate,
        weight_decay                = global_cfg.WEIGHT_DECAY,
        warmup_ratio                = global_cfg.WARMUP_RATIO,
        lr_scheduler_type           = global_cfg.LR_SCHEDULER,
        fp16                        = global_cfg.FP16,
        bf16                        = global_cfg.BF16,
        logging_steps               = global_cfg.LOGGING_STEPS,
        eval_steps                  = global_cfg.EVAL_STEPS,
        save_steps                  = global_cfg.SAVE_STEPS,
        evaluation_strategy         = "steps",
        save_strategy               = "steps",
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        save_total_limit            = 2,
        report_to                   = "none",
        run_name                    = cfg.experiment_id,
        dataloader_num_workers      = 0,
    )

    # Loss history callback
    loss_cb = LossHistoryCallback()

    # Custom callback wrapper so HF Trainer can call it
    from transformers import TrainerCallback

    class _LossCBWrapper(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            loss_cb.on_log(args, state, control, logs, **kwargs)

    trainer = Trainer(
        model         = model,
        args          = training_args,
        train_dataset = train_ds,
        eval_dataset  = val_ds,
        data_collator = data_collator,
        callbacks     = [
            _LossCBWrapper(),
            EarlyStoppingCallback(early_stopping_patience=5),
        ],
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    train_result = trainer.train()

    # ── Save adapter weights ──────────────────────────────────────────────────
    adapter_dir = exp_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    elapsed = time.time() - t_start

    # ── Save training log ─────────────────────────────────────────────────────
    log = {
        "experiment_id":   cfg.experiment_id,
        "config": {
            "dataset_size":         cfg.dataset_size,
            "quantization":         cfg.quantization,
            "lora_r":               cfg.lora_r,
            "lora_alpha":           cfg.lora_alpha,
            "lora_dropout":         cfg.lora_dropout,
            "target_modules":       cfg.target_modules_label,
            "learning_rate":        cfg.learning_rate,
            "num_epochs":           cfg.num_epochs,
        },
        "train_losses":    loss_cb.train_losses,
        "eval_losses":     loss_cb.eval_losses,
        "train_runtime_s": elapsed,
        "train_samples":   len(train_ds),
        "val_samples":     len(val_ds),
        "metrics":         train_result.metrics,
    }
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    # Mark done
    done_marker.touch()

    print(f"\n  ✓ Experiment done in {elapsed:.0f}s  →  {exp_dir}")
    return log


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_id", type=str, required=True,
        help="Experiment ID string matching one config in experiment_config.py"
    )
    args = parser.parse_args()

    import experiment_config as gcfg
    all_exps = gcfg.get_all_experiments()
    matches  = [e for e in all_exps if e.experiment_id == args.experiment_id]
    if not matches:
        raise ValueError(f"No experiment found with id: {args.experiment_id}")
    run_experiment(matches[0], gcfg)

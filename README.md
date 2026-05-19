# Step 2: QLoRA Fine-Tuning Pipeline

## Directory Structure

```
step2_finetune/
├── experiment_config.py   ← ALL sweep variables — edit this first
├── sim_dataset.py         ← dataset loader + tokenisation + loss masking
├── finetune.py            ← core training engine (one experiment)
├── run_experiments.py     ← orchestrator (runs all combinations)
├── plot_training.py       ← learning curve / comparison plots
├── requirements.txt       ← pip dependencies
└── experiments/           ← created at runtime
    └── <experiment_id>/
        ├── checkpoints/   ← HF Trainer checkpoints
        ├── adapter/       ← saved LoRA adapter weights + tokenizer
        ├── training_log.json
        └── DONE           ← marker file (exists = finished)
```

---

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU required** for `4bit` / `8bit` quantization (bitsandbytes).
> Minimum 8 GB VRAM for 3B model with 4-bit; 16 GB for 8-bit.

---

## 2. Configure your sweep

Open `experiment_config.py` and check/edit:

| Setting | What it controls |
|---|---|
| `BASE_MODEL_ID` | HuggingFace model (default: `meta-llama/Llama-3.2-3B`) |
| `HF_TOKEN` | Your HF token (or `None` if already logged in) |
| `DATASET_DIR` | Path to Step 1 output folder |
| `OUTPUT_DIR` | Where experiments are saved |
| `DATASET_SIZES` | `[10, 100, 500]` — questions per model in training |
| `QUANTIZATIONS` | `["4bit", "8bit"]` — add `"none"` for fp16 full |
| `LORA_RANKS` | `[4, 8, 16, 32]` |
| `LORA_ALPHAS` | `["auto_2r"]` = alpha always 2×r, or list of ints |
| `LORA_DROPOUTS` | `[0.05, 0.1]` |
| `TARGET_MODULES_OPTIONS` | `["attention", "mlp", "both"]` |
| `LEARNING_RATES` | `[1e-4, 2e-4, 5e-4]` |
| `NUM_EPOCHS` | `[3]` |
| `PER_DEVICE_BATCH_SIZE` | Lower to `2` or `1` if you get OOM |
| `FP16 / BF16` | `FP16=True` for most GPUs; `BF16=True` for A100/4090 |

### How many experiments?

With the defaults:
- 3 dataset sizes × 2 quantizations × 4 ranks × 1 alpha × 2 dropouts × 3 target modules × 3 LRs × 1 epoch = **432 experiments**

To reduce, comment out values in any list, e.g.:
```python
LORA_RANKS = [8]          # hold r fixed
LORA_DROPOUTS = [0.05]    # hold dropout fixed
```

---

## 3. Preview the experiment grid

```bash
# Print count and first/last IDs
python experiment_config.py

# Print full table (all 432 rows)
python run_experiments.py --list
```

---

## 4. Run the experiments

```bash
# Run everything (skips already-done experiments automatically)
python run_experiments.py --run

# Resume from experiment #50 (e.g. after a crash)
python run_experiments.py --run --start 50

# Run specific experiment IDs only
python run_experiments.py --run --ids ds10_q4bit_r8_a16_do0.05_tmattention_lr2e-04_ep3

# Check status (done / error / pending counts)
python run_experiments.py --status
```

Each finished experiment writes:
- `experiments/<id>/training_log.json` — full loss history + config
- `experiments/<id>/adapter/` — LoRA weights for inference in Step 3

---

## 5. Plot learning curves

```bash
# One experiment
python plot_training.py --exp_id ds10_q4bit_r8_a16_do0.05_tmattention_lr2e-04_ep3

# All completed experiments (one PNG each)
python plot_training.py --all

# Overlay multiple for comparison
python plot_training.py --compare ds10_q4bit_r8_a16_do0.05_tmattention_lr2e-04_ep3 \
                                  ds100_q4bit_r8_a16_do0.05_tmattention_lr2e-04_ep3 \
                                  ds500_q4bit_r8_a16_do0.05_tmattention_lr2e-04_ep3

# Summary plots (heatmaps, dataset-size bar chart)
python plot_training.py --summary
```

Plots are saved to `experiments/plots/`.

---

## 6. What each file produces

| File | Content |
|---|---|
| `training_log.json` | `config`, `train_losses[]`, `eval_losses[]`, `metrics`, `runtime_s` |
| `adapter/` | PEFT adapter weights — load with `PeftModel.from_pretrained()` in Step 3 |
| `plots/*_learning_curves.png` | Train + val loss vs steps (raw + smoothed) |
| `plots/compare_*.png` | Overlay of val loss for multiple experiments |
| `plots/heatmap_lr_r_*.png` | Final val loss heatmap over LR × LoRA-r |
| `plots/dataset_size_effect.png` | Bar chart: mean val loss per dataset size |

---

## Notes

- **Interrupt-safe**: every completed experiment writes a `DONE` file. Re-running always skips done experiments.
- **Error-safe**: errors write an `ERROR` file + `error_log.json` and continue to the next experiment.
- **Loss masking**: only the answer tokens contribute to the loss (the simulation question is masked). This teaches the model to predict numeric outputs, not recite the question.
- **Step 3 compatibility**: the `adapter/` folder from each experiment is directly loadable by `evaluate.py` in Step 3 using `PeftModel.from_pretrained(base_model, adapter_path)`.

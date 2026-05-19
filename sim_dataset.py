"""
sim_dataset.py
==============
Loads the SimPy Q&A dataset produced by Step 1 and prepares it for
causal-language-model fine-tuning.

Each record is converted to a single text string:

    ### Question:
    In a bank simulation with ...?

    ### Answer:
    0.5947

The model is trained to predict the Answer token(s) only (loss masking
on the Question portion).
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional

from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Prompt formatting
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """\
### Simulation Question:
{question}

### Answer:
{answer}"""


def format_prompt(question: str, answer: Optional[float] = None) -> str:
    """
    Return the full prompt string.
    If answer is None (inference mode), return only the question portion
    so the model can complete it.
    """
    if answer is None:
        return f"### Simulation Question:\n{question}\n\n### Answer:\n"
    return PROMPT_TEMPLATE.format(question=question, answer=f"{answer:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Raw JSON loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_json(path: str) -> List[Dict]:
    with open(path, "r") as f:
        return json.load(f)


def subsample_by_model(
    records: List[Dict],
    n_per_model: int,
    sim_models: List[str],
    seed: int = 42,
) -> List[Dict]:
    """
    From the training records, keep at most n_per_model examples
    per simulation model (stratified sub-sample).
    """
    rng = random.Random(seed)
    result = []
    for model in sim_models:
        subset = [r for r in records if r["model"] == model]
        rng.shuffle(subset)
        result.extend(subset[:n_per_model])
    rng.shuffle(result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace-compatible Dataset classes
# ─────────────────────────────────────────────────────────────────────────────

class SimDataset(Dataset):
    """
    PyTorch Dataset that tokenises SimPy Q&A records for causal LM training.

    Loss masking: only the answer tokens contribute to the loss.
    The question / prompt tokens are masked with label = -100.
    """

    def __init__(
        self,
        records: List[Dict],
        tokenizer,
        max_length: int = 256,
    ):
        self.records    = records
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        question = rec["question"]
        answer   = rec["answer"]

        full_text   = format_prompt(question, answer)
        prompt_only = format_prompt(question, answer=None)

        # Tokenise the full text
        full_enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenise prompt only (to know where the answer starts)
        prompt_enc = self.tokenizer(
            prompt_only,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids      = full_enc["input_ids"].squeeze(0)
        attention_mask = full_enc["attention_mask"].squeeze(0)

        # Build labels: -100 for prompt tokens, real ids for answer tokens
        labels = input_ids.clone()
        prompt_len = prompt_enc["input_ids"].shape[1]
        labels[:prompt_len] = -100   # mask the question / prompt portion

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ─────────────────────────────────────────────────────────────────────────────

def build_datasets(
    dataset_dir: str,
    tokenizer,
    dataset_size_per_model: int,
    sim_models: List[str],
    max_length: int = 256,
    seed: int = 42,
):
    """
    Load train / validation / test JSON files, apply sub-sampling
    to the training set, and return three SimDataset objects.

    Returns: (train_ds, val_ds, test_ds, raw_test_records)
    """
    base = Path(dataset_dir)

    train_raw = load_json(base / "dataset_train.json")
    val_raw   = load_json(base / "dataset_validation.json")
    test_raw  = load_json(base / "dataset_test.json")

    # Filter to requested simulation models
    train_raw = [r for r in train_raw if r["model"] in sim_models]
    val_raw   = [r for r in val_raw   if r["model"] in sim_models]
    test_raw  = [r for r in test_raw  if r["model"] in sim_models]

    # Sub-sample training set
    train_raw = subsample_by_model(train_raw, dataset_size_per_model, sim_models, seed)

    # Validation: also sub-sample proportionally (20% of train size, min 10)
    val_size = max(10, dataset_size_per_model // 5)
    val_raw  = subsample_by_model(val_raw, val_size, sim_models, seed)

    train_ds = SimDataset(train_raw, tokenizer, max_length)
    val_ds   = SimDataset(val_raw,   tokenizer, max_length)
    test_ds  = SimDataset(test_raw,  tokenizer, max_length)

    print(f"  Dataset sizes  → train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")
    return train_ds, val_ds, test_ds, test_raw


if __name__ == "__main__":
    # Quick sanity check (no GPU needed)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    import sys; sys.path.insert(0, "..")
    from experiment_config import SIM_MODELS, DATASET_DIR

    tr, va, te, raw = build_datasets(
        DATASET_DIR, tok, dataset_size_per_model=10,
        sim_models=SIM_MODELS, max_length=128,
    )
    sample = tr[0]
    print("input_ids shape :", sample["input_ids"].shape)
    print("labels shape     :", sample["labels"].shape)
    print("Decoded prompt   :")
    print(tok.decode(sample["input_ids"]))

# -*- coding: utf-8 -*-
"""
MAT180 Experiment 1: AdamW vs SophiaG on Tiny GPT (OpenWebText)

This script compares AdamW and SophiaG optimizers when training a Tiny GPT-2 model
on a subset of the OpenWebText dataset for language modeling.

Originally run in Google Colab. For local execution, adapt paths (e.g., sys.path)
and run pip/git commands manually in terminal.
"""

# ------------------------------------------------------------------------------
# Section 1: Environment Check - GPU availability and PyTorch version
# ------------------------------------------------------------------------------

import numpy as np
import random
import time
import torch

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

# ------------------------------------------------------------------------------
# Section 2: Install Dependencies (run once in Colab/terminal)
# For Colab: uncomment and run. For local: run in terminal.
# ------------------------------------------------------------------------------

# !pip install transformers datasets tiktoken wandb --upgrade
# !pip install torch torchvision matplotlib --upgrade

# ------------------------------------------------------------------------------
# Section 3: Clone Sophia Optimizer Repo (run once)
# Sophia: Second-order Clipped Stochastic Optimization (Liu et al.)
# Citation: https://github.com/Liuhong99/Sophia
# For Colab: uncomment. For local: run "git clone https://github.com/Liuhong99/Sophia.git"
# ------------------------------------------------------------------------------

# !rm -rf Sophia  # Run if "fatal: destination path 'Sophia' already exists"
# !git clone https://github.com/Liuhong99/Sophia.git

# ------------------------------------------------------------------------------
# Section 4: Import Libraries
# ------------------------------------------------------------------------------

import sys
import os
# Add Sophia: use ./Sophia when run from project root; /content/Sophia for Colab
_sophia_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Sophia")
if os.path.exists(_sophia_path):
    sys.path.insert(0, _sophia_path)
else:
    sys.path.append("/content/Sophia")  # Colab fallback
from sophia import SophiaG
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# Section 5: Device and Training Hyperparameters
# ------------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Adjustable hyperparameters for language model training
BATCH_SIZE = 16       # Number of sequences per batch
LR = 2e-4             # Learning rate for both optimizers
MAX_STEPS = 1000      # Total training steps (step-based, not epoch-based)
BLOCK_SIZE = 64       # Context length (tokens per sequence)

# ------------------------------------------------------------------------------
# Section 6: Set Random Seeds for Reproducibility
# ------------------------------------------------------------------------------

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ------------------------------------------------------------------------------
# Section 7: Load OpenWebText Dataset (subset for faster experimentation)
# ------------------------------------------------------------------------------

dataset = load_dataset("openwebtext", split="train[:10000]")

# ------------------------------------------------------------------------------
# Section 8: Tokenization - encode text into GPT-2 token IDs
# ------------------------------------------------------------------------------

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def tokenize_function(example):
    """Convert raw text to token IDs using GPT-2 tokenizer."""
    return tokenizer(example["text"])

tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# ------------------------------------------------------------------------------
# Section 9: Create Causal Language Modeling Blocks
# Chunk token sequences into fixed-length blocks for training
# ------------------------------------------------------------------------------

def group_texts(examples):
    """
    Group tokenized texts into fixed-size blocks for causal LM.
    Concatenates all texts, truncates to multiple of BLOCK_SIZE, and splits.
    Labels = input_ids (next-token prediction).
    """

    # Concatenate all token lists in the batch
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}

    # Truncate so total length is divisible by BLOCK_SIZE (drop remainder)
    total_length = len(concatenated_examples["input_ids"])
    total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE

    # Split each column into contiguous chunks of size BLOCK_SIZE
    result = {
        k: [t[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
        for k, t in concatenated_examples.items()
        }

    # For causal LM, labels are the same as input_ids (predict next token)
    result["labels"] = result["input_ids"].copy()

    return result


# Apply grouping to create the final LM dataset
lm_dataset = tokenized.map(
    group_texts,
    batched=True,
    remove_columns=tokenized.column_names
    )


# Split into train (95%) and validation (5%) sets
lm_dataset = lm_dataset.train_test_split(test_size=0.05, seed=SEED)

train_dataset = lm_dataset["train"]
val_dataset = lm_dataset["test"]

# Create DataLoaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ------------------------------------------------------------------------------
# Section 10: Define Tiny GPT-2 Model
# Small architecture for fast experimentation
# ------------------------------------------------------------------------------

config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=BLOCK_SIZE,
    n_embd=128,
    n_layer=2,
    n_head=2,
)

def create_model():
    """Create and return a Tiny GPT-2 LM model on the specified device."""
    return GPT2LMHeadModel(config).to(device)

# ------------------------------------------------------------------------------
# Section 11: Evaluation Function - compute average validation loss
# ------------------------------------------------------------------------------

def evaluate(model):
    model.eval()
    total_loss = 0
    count = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = torch.tensor(batch["input_ids"]).to(device)
            labels = torch.tensor(batch["labels"]).to(device)

            outputs = model(input_ids, labels=labels)
            total_loss += outputs.loss.item()
            count += 1

    model.train()
    return total_loss / count

# ------------------------------------------------------------------------------
# Section 12: Training Function (step-based, not epoch-based)
# Supports both AdamW and SophiaG optimizers
# ------------------------------------------------------------------------------

def train_model(optimizer_name):
    """
    Train Tiny GPT-2 for MAX_STEPS with either 'adamw' or 'sophia' optimizer.
    Returns (train_losses, val_losses, training_time).
    """
    model = create_model()

    if optimizer_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=LR)
    else:
        optimizer = SophiaG(
            model.parameters(),
            lr=LR,
            betas=(0.965, 0.99),
            rho=0.03,
            weight_decay=0.1,
        )

    train_losses = []
    val_losses = []

    step = 0
    start_time = time.time()

    model.train()

    while step < MAX_STEPS:
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, labels=labels)
            loss = outputs.loss

            loss.backward()

            if optimizer_name == "adamw":
                optimizer.step()
            else:
                optimizer.step(bs=BATCH_SIZE)

            optimizer.zero_grad()

            train_losses.append(loss.item())
            step += 1

            if step % 200 == 0:
                val_loss = evaluate(model)
                val_losses.append(val_loss)
                print(f"{optimizer_name} Step {step} | Val Loss: {val_loss:.4f}")

            if step >= MAX_STEPS:
                break

    training_time = time.time() - start_time
    return train_losses, val_losses, training_time

def collate_fn(batch):
    """
    Custom collate: batch is a list of dicts with 'input_ids' and 'labels'.
    Stack them into tensors for the DataLoader.
    """
    input_ids = torch.stack([torch.tensor(x["input_ids"]) for x in batch])
    labels = torch.stack([torch.tensor(x["labels"]) for x in batch])
    return {"input_ids": input_ids, "labels": labels}

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ------------------------------------------------------------------------------
# Section 13: Train Both Optimizers (AdamW and SophiaG)
# ------------------------------------------------------------------------------

print("Training AdamW...")
adam_train, adam_val, adam_time = train_model("adamw")  # train_loss, val_loss, training_time

print("Training SophiaG...")
sophia_train, sophia_val, sophia_time = train_model("sophia")

# ------------------------------------------------------------------------------
# Section 14: Plot Results - validation loss and training loss comparison
# ------------------------------------------------------------------------------

steps = [200*(i+1) for i in range(len(adam_val))]

plt.figure(figsize=(10,5))
plt.plot(steps, adam_val, label="AdamW")
plt.plot(steps, sophia_val, label="Sophia")
plt.xlabel("Training Steps")
plt.ylabel("Validation Loss")
plt.title("AdamW vs Sophia (Tiny GPT on OpenWebText)")
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(range(1, len(adam_train)+1), adam_train, label="AdamW Train Loss")
plt.plot(range(1, len(sophia_train)+1), sophia_train, label="SophiaG Train Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title(f"Train Loss Comparison (Batch={BATCH_SIZE})")
plt.legend()
plt.show()

# ==============================================================================
# EXPERIMENT 2: Sophia rho Hyperparameter Sweep on OpenWebText + Tiny GPT-2
# Examines how the clipping threshold (rho) affects val_loss and clip proportion
# Reuses train_loader, val_loader, create_model, evaluate from Experiment 1
# ==============================================================================

LOG_EVERY  = 100   # Log clip proportion every N steps
EVAL_EVERY = 200   # Evaluate on validation set every N steps (same as Exp1)


def set_seed(seed=42):
    """Set random seeds for reproducibility (resets for each rho run)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def compute_clip_prop(optimizer, rho=None, bs=1, eps=1e-15, gamma=1.0):
    """
    Compute the proportion of coordinates that are clipped in the current SophiaG step.

    SophiaG update: ratio_raw = |m| / (rho * bs * h + eps), ratio = ratio_raw.clamp(max=1).
    Clipping condition: ratio_raw > gamma (gamma=1 in standard SophiaG).
    Returns the fraction of parameters where clipping occurred.
    """
    total_elems = 0
    clip_elems = 0

    if rho is None:
        rho = optimizer.param_groups[0].get("rho", None)
    if rho is None:
        return None

    for p, st in optimizer.state.items():
        if not isinstance(st, dict):
            continue
        if ("exp_avg" not in st) or ("hessian" not in st):
            continue

        m = st["exp_avg"]
        h = st["hessian"]

        if (m is None) or (h is None):
            continue
        if (not torch.is_tensor(m)) or (not torch.is_tensor(h)):
            continue
        if m.numel() == 0:
            continue

        denom = rho * bs * h + eps
        ratio_raw = m.abs() / denom

        total_elems += ratio_raw.numel()
        clip_elems += (ratio_raw > gamma).sum().item()

    if total_elems == 0:
        return None

    clip_prop = clip_elems / total_elems
    return clip_prop


def train_sophia_with_rho(
    rho,
    seed=42,
    max_steps=MAX_STEPS,
    log_every=LOG_EVERY,
    eval_every=EVAL_EVERY,
    gamma=1.0,
):
    """
    Train Tiny GPT-2 on OpenWebText with SophiaG using a given rho (clipping threshold).
    Reuses train_loader, val_loader, create_model, evaluate from Experiment 1.
    Returns dict with val_steps, val_losses, and clip_props.
    """
    set_seed(seed)
    model = create_model()

    optimizer = SophiaG(
        model.parameters(),
        lr=LR,
        betas=(0.965, 0.99),
        rho=rho,
        weight_decay=0.1,
    )

    print("optimizer group rho =", optimizer.param_groups[0].get("rho", None))

    steps = 0
    val_steps = []
    val_losses = []
    clip_props = []

    train_iter = iter(train_loader)

    model.train()
    while steps < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.update_hessian()

        optimizer.step(bs=BATCH_SIZE)
        steps += 1

        if steps % log_every == 0:
            clip_prop = compute_clip_prop(optimizer, rho=rho, bs=BATCH_SIZE, gamma=gamma)
            clip_props.append(clip_prop)

        if steps % eval_every == 0:
            val_loss = evaluate(model)
            val_steps.append(steps)
            val_losses.append(val_loss)
            print(f"[rho={rho}] step {steps} | val_loss={val_loss:.4f}")

    return {
        "val_steps": val_steps,
        "val_losses": val_losses,
        "clip_props": clip_props,
    }

# rho sweep
RHO_LIST = [1e-4, 1e-2, 1e-0]
SEED = SEED

results = {}  # rho -> dict

for rho in RHO_LIST:
    print("=" * 40)
    print(f"Run with rho={rho}")
    res = train_sophia_with_rho(
        rho=rho,
        seed=SEED,
        max_steps=MAX_STEPS,
        log_every=LOG_EVERY,
        eval_every=EVAL_EVERY,
    )
    results[rho] = res

# Select best rho by minimum final validation loss
best_rho = min(
    RHO_LIST,
    key=lambda r: results[r]["val_losses"][-1] if results[r]["val_losses"] else float("inf"),
)
print("\nBest rho by final val_loss:", best_rho)

RHO_NOCLIP = 1e6
print("\nRun approximate No-Clip with very large rho =", RHO_NOCLIP)
res_noclip = train_sophia_with_rho(
    rho=RHO_NOCLIP,
    seed=SEED,
    max_steps=MAX_STEPS,
    log_every=LOG_EVERY,
    eval_every=EVAL_EVERY,
)

final_val_nc = res_noclip["val_losses"][-1] if res_noclip["val_losses"] else float("inf")
cp_nc = [c for c in res_noclip["clip_props"] if c is not None]
mean_clip_nc = np.mean(cp_nc) if cp_nc else np.nan
print(f"No-Clip approx (rho={RHO_NOCLIP}): final_val_loss={final_val_nc:.4f}, mean_clip_prop={mean_clip_nc:.4f}")

# Figure 1: Validation loss vs step for different rho values
plt.figure(figsize=(7, 4))
for rho in RHO_LIST:
    st = results[rho]
    plt.plot(st["val_steps"], st["val_losses"], marker="o", label=f"rho={rho}")
plt.xlabel("step")
plt.ylabel("val_loss")
plt.title("Exp2: val_loss vs step (different rho) - Tiny GPT on OpenWebText")
plt.legend()
plt.tight_layout()
plt.show()

# Figure 2: clip_prop curves (multiple rho lines)
plt.figure(figsize=(7, 4))
for rho in RHO_LIST:
    cp = results[rho]["clip_props"]
    steps = np.arange(len(cp)) * LOG_EVERY
    plt.plot(steps, cp, marker="o", label=f"rho={rho}")
plt.xlabel("step")
plt.ylabel("clipping proportion")
plt.title("Exp2: clipping proportion vs step - Tiny GPT on OpenWebText")
plt.legend()
plt.tight_layout()
plt.show()

# Summary table: Final val_loss and average clip_prop for each rho
print("rho\tfinal_val_loss\tmean_clip_prop")
print("-" * 40)
for rho in RHO_LIST:
    st = results[rho]
    final_val = st["val_losses"][-1] if st["val_losses"] else np.nan
    cp = [c for c in st["clip_props"] if c is not None]
    mean_clip = np.mean(cp) if cp else np.nan
    print(f"{rho}\t{final_val:.4f}\t\t{mean_clip:.4f}")
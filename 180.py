# -*- coding: utf-8 -*-
"""
MAT180: AdamW vs SophiaG Optimizer Comparison on MNIST

This script compares the AdamW and SophiaG optimizers when training a simple MLP
on the MNIST digit classification dataset. Includes Experiment 2: Sophia rho
hyperparameter sweep.

Originally run in Google Colab. For local execution, adapt paths and run
pip/git commands manually.
"""

# ------------------------------------------------------------------------------
# Section 1: Environment Check - GPU and PyTorch version
# ------------------------------------------------------------------------------

import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

# ------------------------------------------------------------------------------
# Section 2: Install Dependencies (run once)
# For Colab: uncomment the lines below. For local: run in terminal.
# ------------------------------------------------------------------------------

# !pip install transformers datasets tiktoken wandb --upgrade
# !pip install torch torchvision matplotlib --upgrade

# ------------------------------------------------------------------------------
# Section 3: Clone Sophia Optimizer Repo (run once)
# Citation: https://github.com/Liuhong99/Sophia
# For Colab: uncomment. For local: run "git clone https://github.com/Liuhong99/Sophia.git"
# ------------------------------------------------------------------------------

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
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# Section 5: Device and Training Hyperparameters
# ------------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Adjustable hyperparameters
BATCH_SIZE = 512   # Batch size for training
EPOCHS = 10        # Number of epochs (only 2 used in train_model)
LR = 1e-3          # Learning rate

# ------------------------------------------------------------------------------
# Section 6: Prepare MNIST Dataset
# ------------------------------------------------------------------------------

transform = transforms.ToTensor()
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

# ------------------------------------------------------------------------------
# Section 7: Define Simple MLP Model (784 -> 256 -> 10)
# ------------------------------------------------------------------------------

# Simple MLP for MNIST digit classification
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    def forward(self, x):
        return self.fc(x)

# ------------------------------------------------------------------------------
# Section 8: Training Function - supports AdamW and SophiaG
# ------------------------------------------------------------------------------

def train_model(optimizer_name):
    """
    Train the MLP for 2 epochs with either 'adamw' or 'sophia' optimizer.
    Returns list of training losses per iteration.
    """
    model = Net().to(device)

    if optimizer_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    else:
        optimizer = SophiaG(model.parameters(), lr=1e-3,
                            betas=(0.965, 0.99), rho=0.01,
                            weight_decay=0.01)

    criterion = nn.CrossEntropyLoss()
    losses = []

    for epoch in range(2):  # only 2 epochs
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()

            if optimizer_name == "adamw":
                optimizer.step()
            else:
                optimizer.step(bs=256)

            optimizer.zero_grad()
            losses.append(loss.item())

    return losses

# ------------------------------------------------------------------------------
# Section 9: Train Both Optimizers and Plot Loss Comparison
# ------------------------------------------------------------------------------

adam_losses = train_model("adamw")
sophia_losses = train_model("sophia")

# Plot loss curves for AdamW vs SophiaG
plt.figure(figsize=(10,5))
plt.plot(adam_losses, label="AdamW")
plt.plot(sophia_losses, label="SophiaG")
plt.legend()
plt.title(f"Loss Comparison (batch={BATCH_SIZE}, epochs={EPOCHS})")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

# ==============================================================================
# EXPERIMENT 2: Sophia rho Hyperparameter Sweep on MNIST
# Studies how rho (clipping threshold) affects validation loss and clip proportion
# ==============================================================================

import sys
sys.path.append("/content/Sophia")

from sophia import SophiaG

# Basic imports and device setup
import random, numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# Hyperparameters and MNIST data configuration
BATCH_SIZE = 512   # Batch size
LR = 1e-3          # Learning rate

MAX_STEPS  = 2000  # Total training steps
LOG_EVERY  = 100   # Log clip proportion every N steps
EVAL_EVERY = 200   # Evaluate validation set every N steps
SEED       = 42    # Random seed for reproducibility

transform = transforms.ToTensor()
trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
testset  = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
testloader  = torch.utils.data.DataLoader(testset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# MLP: 784 -> 256 -> 128 -> 10
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.fc(x)


# Utility: set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def eval_model(model, loader):
    """
    Evaluate model on the given data loader.
    Returns (average_loss, accuracy).
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    model.train()
    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


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
    Train MLP with SophiaG using a given rho (clipping threshold).
    Returns dict with val_steps, val_losses, and clip_props.
    """
    set_seed(seed)
    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = SophiaG(
        model.parameters(),
        lr=LR,
        betas=(0.965, 0.99),
        rho=rho,
        weight_decay=0.01,
    )

    print("optimizer group rho =", optimizer.param_groups[0].get("rho", None))

    steps = 0
    val_steps = []
    val_losses = []
    clip_props = []

    train_iter = iter(trainloader)

    while steps < max_steps:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(trainloader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.update_hessian()

        optimizer.step(bs=BATCH_SIZE)
        steps += 1

        if steps % log_every == 0:
            clip_prop = compute_clip_prop(optimizer, rho=rho, bs=BATCH_SIZE, gamma=gamma)
            clip_props.append(clip_prop)

        if steps % eval_every == 0:
            val_loss, _ = eval_model(model, testloader)
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
plt.title("Exp2: val_loss vs step (different rho)")
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
plt.title("Exp2: clipping proportion vs step")
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
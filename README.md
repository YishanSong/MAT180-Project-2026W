# MAT180: AdamW vs SophiaG Optimizer Comparison

A baseline model/pipeline comparing **AdamW** and **SophiaG** optimizers on:
1. **180.py** — MNIST digit classification with MLP; includes Sophia rho hyperparameter sweep
2. **180exp1_openwebtext.py** — Tiny GPT-2 language modeling on OpenWebText; AdamW vs SophiaG

## Setup Instructions

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended; CPU also supported)

### 1. Clone the Sophia optimizer repository

```bash
git clone https://github.com/Liuhong99/Sophia.git
```

### 2. Install dependencies

```bash
pip install torch torchvision matplotlib
pip install transformers datasets tiktoken
```

For `180exp1_openwebtext.py` only:

```bash
pip install transformers datasets tiktoken
```

### 3. Set Python path

Ensure the `Sophia` directory is in your Python path. In code:

```python
import sys
sys.path.append("./Sophia")  # or "/content/Sophia" in Colab
```

## How to Run Training / Evaluation

### 180.py (MNIST)

```bash
python 180.py
```

- **Experiment 1:** Trains an MLP on MNIST for 2 epochs with AdamW and SophiaG, then plots loss comparison.
- **Experiment 2:** Runs SophiaG with different `rho` values (1e-4, 1e-2, 1e0) and plots val_loss and clipping proportion.

### 180exp1_openwebtext.py (OpenWebText + Tiny GPT-2)

```bash
python 180exp1_openwebtext.py
```

Note: In Colab, uncomment the `!pip` and `!git` lines in the script. For local runs, those commands are commented out—run them in your terminal instead, and set `sys.path.append("./Sophia")` (or your local Sophia path).

- **Experiment 1:** Loads OpenWebText subset, trains Tiny GPT-2 with AdamW and SophiaG for 1000 steps, plots validation and training loss.
- **Experiment 2:** Same Sophia rho sweep as in 180.py on MNIST.

## Where Outputs / Results Go

| Output | Location |
|--------|----------|
| **Loss curves (plots)** | Displayed in-place when running (e.g., `plt.show()`). To save, add `plt.savefig("output.png")` before `plt.show()`. |
| **MNIST data** | Downloaded to `./data/` on first run |
| **Console logs** | Printed to terminal (val_loss, step count, best rho, clip proportion) |

Suggested additions for saving outputs:

```python
# Save figures
plt.savefig("results/loss_comparison.png")
plt.savefig("results/val_loss_rho.png")
plt.savefig("results/clip_prop_rho.png")
```

Create a `results/` directory and add the above before `plt.show()` if you want files saved.

## Citations for Reused Resources

- **Sophia Optimizer:** [Liu et al., Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training](https://arxiv.org/abs/2305.14342) — [GitHub](https://github.com/Liuhong99/Sophia)
- **Hugging Face Transformers / Datasets:** [https://huggingface.co/docs](https://huggingface.co/docs)
- **OpenWebText:** [https://huggingface.co/datasets/openwebtext](https://huggingface.co/datasets/openwebtext)
- **GPT-2:** Radford et al., "Language Models are Unsupervised Multitask Learners"
- **MNIST:** [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

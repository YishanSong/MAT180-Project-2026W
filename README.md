# MAT180: AdamW vs SophiaG Optimizer Comparison

A baseline model/pipeline comparing **AdamW** and **SophiaG** optimizers. Both scripts run **Experiment 1** and **Experiment 2**; the difference is the dataset:

| File | Dataset / Task |
|------|----------------|
| **180.py** | MNIST digit classification (MLP) |
| **180exp1_openwebtext.py** | OpenWebText + Tiny GPT-2 (language modeling) |

Each file contains:
- **Experiment 1:** AdamW vs SophiaG optimizer comparison
- **Experiment 2:** Sophia rho hyperparameter sweep

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

Runs both experiments on MNIST:
- **Experiment 1:** AdamW vs SophiaG on MLP for 2 epochs; plots loss comparison.
- **Experiment 2:** SophiaG rho sweep (1e-4, 1e-2, 1e0); plots val_loss and clipping proportion.

### 180exp1_openwebtext.py (OpenWebText + Tiny GPT-2)

```bash
python 180exp1_openwebtext.py
```

Runs both experiments on OpenWebText/Tiny GPT-2:
- **Experiment 1:** AdamW vs SophiaG on Tiny GPT-2 for 1000 steps; plots validation and training loss.
- **Experiment 2:** SophiaG rho sweep on OpenWebText + Tiny GPT-2 (same setup as Experiment 1).

Note: In Colab, uncomment the `!pip` and `!git` lines. For local runs, run those commands in your terminal instead.

## Where Outputs / Results Go

| Output | Location |
|--------|----------|
| **Loss curves (plots)** | Displayed in-place when running (e.g., `plt.show()`). To save, add `plt.savefig("output.png")` before `plt.show()`. |
| **MNIST data** | Downloaded to `./data/` on first run (180.py only) |
| **OpenWebText data** | Loaded from Hugging Face Datasets on first run (180exp1_openwebtext) |
| **Console logs** | Printed to terminal (val_loss, step count, best rho, clip proportion) |

Suggested additions for saving outputs:

```python
# Save figures
plt.savefig("results/loss_comparison.png")
plt.savefig("results/val_loss_rho.png")
plt.savefig("results/clip_prop_rho.png")
```

Create a `results/` directory and add the above before `plt.show()` if you want files saved.

## Implementation Notes

- **Experiment 2 (Sophia rho sweep):** SophiaG requires `optimizer.update_hessian()` to be called after `loss.backward()` and before `optimizer.step()`, otherwise the Hessian stays at zeros and rho has no effect on the update.

## Citations for Reused Resources

- **Sophia Optimizer:** [Liu et al., Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training](https://arxiv.org/abs/2305.14342) — [GitHub](https://github.com/Liuhong99/Sophia)
- **Hugging Face Transformers / Datasets:** [https://huggingface.co/docs](https://huggingface.co/docs)
- **OpenWebText:** [https://huggingface.co/datasets/openwebtext](https://huggingface.co/datasets/openwebtext)
- **GPT-2:** Radford et al., "Language Models are Unsupervised Multitask Learners"
- **MNIST:** [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

# CPFN — Conditional Push-Forward Neural Network

Compact, importable implementation of a Conditional Push-Forward Neural Network (CPFN) estimator. 

**Paper:** https://arxiv.org/pdf/2511.14455

## Goals
- Provide a lightweight `CPFN` class for estimating conditional generators.
- Expose a simple API for training and sampling.


## Install

### From PyPI
```bash
pip install cpfn
```

## Quick Usage
```python

import random
import numpy as np
import torch
from cpfn import CPFN

# matplotlib is not a dependency of cpfn — install separately if needed:
#   pip install matplotlib
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Setup
# ---------------------------------------------------------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# ---------------------------------------------------------------------------
# 2. Synthetic Data — Branching Distribution
# ---------------------------------------------------------------------------
# For x < 0.5: single Gaussian branch (mu1).
# For x >= 0.5: equal-weight mixture of two Gaussian branches (mu1, mu2).

def mu1(x):
    return 10 * x * (x - 0.5) * (1.5 - x)

def mu2(x):
    return 10 * x * (x - 0.5) * (0.8 - x)

def noise_std(x):
    return 0.3 * (1.3 - x)

def sample_y(x):
    z = np.random.randn()
    if x < 0.5 or np.random.rand() < 0.5:
        return mu1(x) + z * noise_std(x)
    else:
        return mu2(x) + z * noise_std(x)

def true_conditional_pdf(y, x):
    """Analytic conditional density p(y | x)."""
    s = noise_std(x)
    def gauss(y, m): 
        return np.exp(-0.5 * ((y - m) / s) ** 2) / (np.sqrt(2 * np.pi) * s)
    if x < 0.5:
        return gauss(y, mu1(x))
    return 0.5 * gauss(y, mu1(x)) + 0.5 * gauss(y, mu2(x))


N_TRAIN = 1000
xs = np.random.rand(N_TRAIN)
ys = np.array([sample_y(x) for x in xs])

# ---------------------------------------------------------------------------
# 3. Model Training
# ---------------------------------------------------------------------------

model = CPFN(d=1, q=1, r=20, width=50, hidden_layers=3, delta=1e-15)
model.to(device)

model.fit(xs, ys, epochs=3000, lr=1e-3, m=30, h0=5e-2)
model.freeze()

# ---------------------------------------------------------------------------
# 4. Sample Comparison Plot
# ---------------------------------------------------------------------------

ys_gen = model.sample_conditional(xs, num_samples=1).flatten()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

ax1.scatter(xs, ys, alpha=0.6, s=15, color="steelblue")
ax1.set_title("Ground Truth Samples")
ax1.set_xlabel("x")
ax1.set_ylabel("y")

ax2.scatter(xs, ys_gen, alpha=0.6, s=15, color="darkorange")
ax2.set_title("CPFN Generated Samples")
ax2.set_xlabel("x")

fig.suptitle("Training Data vs. CPFN Samples", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("assets/sample_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------------------------
# 5. Conditional Density Comparison
# ---------------------------------------------------------------------------

ygrid = np.linspace(-1.5, 3.0, 1000)
x_evals = [0.3, 0.7]

fig, axes = plt.subplots(1, len(x_evals), figsize=(5 * len(x_evals), 4), sharey=True)

for ax, x0 in zip(axes, x_evals):
    model_density = np.exp(model.logdensity(x0, ygrid, m=100_000))
    true_density  = true_conditional_pdf(ygrid, x0)

    ax.plot(ygrid, model_density, label="CPFN", color="darkorange", linewidth=1.8)
    ax.fill_between(ygrid, 0, model_density, alpha=0.20, color="darkorange")

    ax.plot(ygrid, true_density, label="True", color="steelblue",
            linestyle="--", linewidth=1.8)
    ax.fill_between(ygrid, 0, true_density, alpha=0.12, color="steelblue")

    ax.set_title(f"p(y | x = {x0:.1f})")
    ax.set_xlabel("y")
    ax.legend()

axes[0].set_ylabel("Density")
fig.suptitle("Conditional Density: CPFN vs. True", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("assets/conditional_density.png", dpi=150, bbox_inches="tight")
plt.show()
```
## Results

**Samples: Training Data vs. CPFN**
![Sample Comparison](sample_comparison.png)

**Conditional Density: CPFN vs. True**
![Conditional Density](conditional_density.png)

## Tests
Run the included pytest smoke test:
```bash
pytest -q
```

## Development
- Source: `src/cpfn/`
- Tests: `tests/`

## License
See `LICENCE` in the repository root.

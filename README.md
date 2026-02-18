# CPFN — Conditional Push-Forward Neural Network

Compact, importable implementation of a Conditional Push-Forward Neural Network (CPFN) estimator. 

**Paper:** https://arxiv.org/pdf/2511.14455

## Goals
- Provide a lightweight `CPFN` class for estimating conditional generators.
- Expose a simple API for training and sampling.

## Install

### From source (development)
```bash
git clone https://github.com/tedescolor/cpfn.git
cd cpfn
pip install -e .
```

### From PyPI (when published)
```bash
pip install cpfn
```

## Quick Usage
```python
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from cpfn import CPFN

# 0. Hardware Selection (CUDA for NVIDIA, MPS for Apple Silicon, or CPU) and Reproducibility
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

SEED = 43

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 1. Define Synthetic Ground Truth (Branching Function)
def true_sample(x):
    z = np.random.randn()  # Gaussian noise
    p = np.random.rand()   # Uniform switch

    # Conditional logic: creates two paths for x > 0.5
    if x < 0.5 or p < 0.5:
        return 10 * x * (x - 0.5) * (1.5 - x) + z * 0.3 * (1.3 - x)
    else:
        return 10 * x * (x - 0.5) * (0.8 - x) + z * 0.3 * (1.3 - x)


# 2. Generate Training Data
ntrain = 500
xs = np.random.rand(ntrain)
ys = np.array([true_sample(x) for x in xs])

# 3. Model Setup & Training
model = CPFN(d=1, q=1, r=20, width=50, hidden_layers=3, delta=1e-15)
model.to(device)
model.fit(xs, ys,
          epochs=3000,
          lr=1e-3,
          m=30,
          h0=5e-2)

model.freeze()

# 4. Inference: Generate 1 sample for every x in training set
# samples shape: (ntrain, 1, 1)
ys_gen = model.sample_conditional(xs, num_samples=1).flatten()

# 5. Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

ax1.scatter(xs, ys, alpha=0.6, s=15, label="Ground Truth")
ax1.set_title("Original Training Data")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
```


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

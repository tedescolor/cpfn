# CPFN — Conditional Push-Forward Neural Network

Compact, importable implementation of a Conditional Push-Forward Neural Network (CPFN) estimator. 

**Paper:** https://arxiv.org/pdf/2511.14455

## Goals
- Provide a lightweight `CPFN` class for estimating conditional generators.
- Expose a simple API for training and sampling.

## Install

### From source (development)
```bash
git clone https://github.com/your-username/cpfn.git
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
from cpfn import CPFN

# device
device = torch.device('cpu')

# toy data (n, d) and (n, q)
xs = torch.rand(100, 1, device=device)
ys = torch.sin(2 * torch.pi * xs)

# create model
model = CPFN(d=1, q=1, r=20, width=50, hidden_layers=3)
model.to(device)

# train (short example)
model.fit(xs, ys, epochs=1000, lr=1e-3, m=30, h0=5e-2)

#freeze the model 
model.freeze()

# sample conditional draws
samples = model.sample_conditional(xs[:10], num_samples=50, seed=42)
print(samples.shape)  # (10, 50, 1)
```

## Plotting Example
```python
import matplotlib.pyplot as plt

xs_np = xs.cpu().numpy().flatten()
ys_np = ys.cpu().numpy().flatten()
samples_np = samples.detach().numpy()  # (n, num_samples, q)

plt.figure(figsize=(6, 4))
plt.plot(xs_np, ys_np, 'k.', label='data')
for j in range(samples_np.shape[1]):
    plt.scatter(xs_np[:10], samples_np[:, j, 0], s=6, alpha=0.2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('CPFN conditional samples')
plt.legend()
plt.show()
```

## Spatial CPFN (2D+)
For spatial regression with Wendland basis embeddings:

```python
import torch
import numpy as np
from cpfn import SpatialCPFN, PhysicsInformedSpatialCPFN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Setup multiresolution Wendland basis
num_basis = [10**2, 19**2, 37**2]
knots_1d = [np.linspace(0, 1, int(np.sqrt(m))) for m in num_basis]

centers_list = []
radii_list = []
for m, ks in zip(num_basis, knots_1d):
    theta = 2.5 / np.sqrt(m)
    k1, k2 = np.meshgrid(ks, ks)
    knots = np.column_stack([k1.ravel(), k2.ravel()])
    centers_list.append(knots)
    radii_list.append(np.full(m, theta))

centers = torch.from_numpy(np.vstack(centers_list)).float().to(device)
radii = torch.from_numpy(np.concatenate(radii_list)).float().to(device)

# Create spatial data (e.g., 100 random 2D locations)
s_train = torch.rand(100, 2, device=device)  # spatial coordinates
y_train = torch.sin(2 * np.pi * s_train[:, :1]).to(device)  # targets

# Train spatial CPFN
model = SpatialCPFN(
    centers=centers,
    radius=radii,
    q=1, r=500, width=100, hidden_layers=3
).to(device)

model.fit(s_train, y_train, epochs=1000, lr=1e-4)
samples = model.sample_conditional(s_train[:10], num_samples=50)

# Or train with physics constraints
pinn_model = PhysicsInformedSpatialCPFN(
    centers=centers,
    radius=radii,
    q=1, r=500, width=100, hidden_layers=3
).to(device)

pinn_model.fit_physics(
    s_train=s_train,
    y_train=y_train,
    epochs=5000,
    lr=1e-4,
    lambda_spde=0.1,
    kappa_init=10.0
)
```

## API (Short)

### Core CPFN
- `CPFN(d, q, ...)` — constructor
- `fit(xs, ys, epochs, lr, m, h0)` — train model  
  - `xs`: `(n, d)`, `ys`: `(n, q)`
- `sample_conditional(x, num_samples, seed)` — returns `(n, num_samples, q)`
- `eps()` — return learned bandwidth(s)
- `freeze()` — set model to eval mode and stop gradients

### Spatial Extensions
- `WendlandEmbedding(centers, radius)` — fixed Wendland basis embedding
- `SpatialCPFN(centers, radius, q, ...)` — CPFN with spatial embedding
  - `fit(s_train, y_train, epochs, lr, ...)` — train on spatial data
  - `sample_conditional(s, num_samples)` — spatial conditional sampling
- `PhysicsInformedSpatialCPFN(...)` — CPFN with SPDE constraints
  - `fit_physics(s_train, y_train, epochs, lambda_spde, ...)` — physics-informed training

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

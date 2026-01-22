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

import math

import torch

from cpfn import CPFN


def test_cpfn_smoke_cpu():
    device = torch.device("cpu")
    torch.manual_seed(0)

    n = 20
    xs = torch.rand(n, 1, device=device)
    ys = torch.sin(2 * math.pi * xs)

    model = CPFN(d=1, q=1, r=4, width=16, hidden_layers=1, learn_eps=True)
    model.to(device)

    # short training to smoke-test functionality
    model.fit(xs, ys, epochs=5, lr=1e-3, m=5, h0=1e-2)

    out = model.sample_conditional(xs[:4], num_samples=3, seed=42)
    assert out.shape == (4, 3, 1)

    eps = model.eps().detach().cpu().numpy()
    assert (eps > 0).all()

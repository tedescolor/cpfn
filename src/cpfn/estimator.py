import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

gelu = nn.GELU()

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_width: int = 50, hidden_layers: int = 3, activation: nn.Module = gelu, final_activation: bool = False):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(d, hidden_width))
            layers.append(activation)
            d = hidden_width
        layers.append(nn.Linear(d, out_dim))
        if(final_activation):
            layers.append(activation)
        self.net = nn.Sequential(*layers)
        self.final_activation = final_activation

    def forward(self, x: torch.Tensor):
        return self.net(x)


class CPFN(nn.Module):
    """Compact CPFN estimator suitable for import and use.

    Usage: from cpfn.estimator import CPFN
           model = CPFN(d=1, q=1)
           model.to(device)
           model.fit(xs, ys, ...)
    """
    def __init__(
        self,
        d: int,
        q: int,
        r: int = 20,
        width: int = 50,
        hidden_layers: int = 3,
        latent_dist: str = "normal",
        learn_eps: bool = True,
        eps_init: float = 5e-2,
        delta: float = 1e-15,
        psi_final_activation: bool = True,
    ):
        super().__init__()
        self.d, self.q, self.r = d, q, r
        self.latent_dist = latent_dist
        self.delta = float(delta)

        self.varphi = MLP(d, r * q, hidden_width=width, hidden_layers=hidden_layers, final_activation=False)
        self.psi = MLP(q, r * q, hidden_width=width, hidden_layers=hidden_layers, final_activation=psi_final_activation)

        eps0 = torch.tensor([eps_init] * q, dtype=torch.float32)
        if learn_eps:
            self._eps_param = nn.Parameter(torch.log(torch.expm1(eps0)))
        else:
            self.register_buffer("_eps_param", torch.log(torch.expm1(eps0)))
            self._eps_param.requires_grad_(False)

    def eps(self) -> torch.Tensor:
        return F.softplus(self._eps_param) + 1e-12

    def _sample_u(self, n: int, m: int, device: torch.device) -> torch.Tensor:
        if self.latent_dist == "uniform":
            return torch.rand(n, m, self.q, device=device)
        return torch.randn(n, m, self.q, device=device)

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        if u.dim() == 2:
            vx = self.varphi(x).view(-1, self.r, self.q)
            vu = self.psi(u).view(-1, self.r, self.q)
            return (vx * vu).sum(dim=1)

        if u.dim() == 3:
            n, m, q = u.shape
            vx = self.varphi(x).view(n, self.r, self.q)
            vu = self.psi(u.reshape(n * m, q)).view(n, m, self.r, self.q)
            y = (vx[:, None, :, :] * vu).sum(dim=2)
            return y

        raise ValueError("u must have shape (n,q) or (n,m,q).")

    def logdensity(self, xs: torch.Tensor, ys : torch.Tensor, m : int = 30, tilted : bool = False):
        delta = self.delta if tilted else 1e-15
        u = self._sample_u(xs.shape[0], m, device=xs.device)
        yhat = self.forward(xs, u)
        resid = (ys[:, None, :] - yhat)
        eps = self.eps()
        zs = residual / eps
        rs = zs.pow(2).sum(dim=-1)
        exponents = rs - -0.5 * self.q * math.log(2.0 * math.pi) - torch.log(eps).sum() - math.log(m) - math.log(delta)
        shape = exponents.shape
        shape[1] = 1
        return torch.logsumexp(torch.cat([exponents, torch.zeros(shape, device = xs.device)], dim=1)) + math.log(delta)        

    def sample_conditional(self, x: torch.Tensor, num_samples: int = 1, seed: Optional[int] = None) -> torch.Tensor:
        if seed is not None:
            g = torch.Generator(device=x.device)
            g.manual_seed(int(seed))
            if self.latent_dist == "uniform":
                u = torch.rand(x.shape[0], num_samples, self.q, generator=g, device=x.device)
            else:
                u = torch.randn(x.shape[0], num_samples, self.q, generator=g, device=x.device)
        else:
            u = self._sample_u(x.shape[0], num_samples, x.device)

        y = self.forward(x, u)
        return y

    def fit(self, xs: torch.Tensor, ys: torch.Tensor, epochs: int = 1000, lr: float = 1e-3, m: int = 30, h0: float = 5e-2):
        device = xs.device
        n = xs.shape[0]

        with torch.no_grad():
            eps0 = torch.tensor([h0] * self.q, device=device, dtype=torch.float32)
            try:
                self._eps_param.copy_(torch.log(torch.expm1(eps0)))
            except Exception:
                pass

        opt = torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=lr)

        pbar = tqdm(range(epochs), desc="Training CPFN")
        for epoch in pbar:
            self.train()
            opt.zero_grad()
            loss = -self.logdensity(xs, ys, m, tilted = True).mean()
            loss.backward()
            opt.step()
            
            # Update progress bar with loss and bandwidth info
            eps_vals = self.eps().detach().cpu().numpy()
            eps_str = ", ".join([f"{e:.2e}" for e in eps_vals])
            pbar.set_postfix({
                "loss": f"{loss.item():.4e}",
                "bandwidth": eps_str
            })

    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)

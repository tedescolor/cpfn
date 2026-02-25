import math
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import copy

gelu = nn.GELU()

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_width: int = 50, hidden_layers: int = 3, activation: nn.Module = gelu, final_activation: bool = False):
        super().__init__()
        layers = []
        d = in_dim
        # Dynamically build hidden layers
        for _ in range(hidden_layers):
            layers.append(nn.Linear(d, hidden_width))
            layers.append(activation)
            d = hidden_width
        # Output layer
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
        d: int,          # Input dimension (x)
        q: int,          # Output dimension (y)
        r: int = 20,     # Rank (latent factor dimension)
        p: Optional[int] = None,  # Latent variable dimension (u); defaults to q if None
        width: int = 50,
        hidden_layers: int = 3,
        latent_dist: str = "normal", # Distribution for latent variable u
        learn_eps: bool = True, # Whether to learn the bandwidth parameter
        eps_init: float = 5e-2,
        delta: float = 1e-15, # Numerical stability constant
        psi_final_activation: bool = True,

    ):
        super().__init__()
        self.d, self.q, self.r = d, q, r
        self.p = p if p is not None else q  # Latent dimension defaults to q
        self.latent_dist = latent_dist
        self.delta = float(delta)
        self._istraining = False

        # Precompute rank scaling factor (1/sqrt(r)) to stabilize output variance across different ranks
        self.register_buffer("rank_scale", torch.tensor(r ** -0.5))

        # Neural networks for the conditional (varphi) and latent (psi) components
        # Outputs are flattened matrices of shape (r * q)
        self.varphi = MLP(d, r * q, hidden_width=width, hidden_layers=hidden_layers, final_activation=False)
        self.psi = MLP(self.p, r * q, hidden_width=width, hidden_layers=hidden_layers, final_activation=psi_final_activation)

        # We use exp() to ensuring positivity
        eps0 = torch.tensor([eps_init] * q, dtype=torch.float32)
        if learn_eps:
            self._eps_param = nn.Parameter(torch.log(eps0))
        else:
            self.register_buffer("_eps_param", torch.log(eps0))
            self._eps_param.requires_grad_(False)

        # Standardization statistics (initialized as None, computed during fit)
        self.register_buffer("x_mean", None)
        self.register_buffer("x_std", None)
        self.register_buffer("y_mean", None)
        self.register_buffer("y_std", None)


    def eps(self) -> torch.Tensor:
        return torch.exp(self._eps_param) + 1e-12

    def _sample_u(self, n: int, m: int, device: torch.device) -> torch.Tensor:
        # Sample latent variable u from defined prior
        if self.latent_dist == "uniform":
            return torch.rand(n, m, self.p, device=device)
        return torch.randn(n, m, self.p, device=device)

    def _standardize_x(self, x: torch.Tensor) -> torch.Tensor:
        """Standardize input x using stored statistics."""
        if self.x_mean is None or self.x_std is None:
            return x
        return (x - self.x_mean) / self.x_std

    def _standardize_y(self, y: torch.Tensor) -> torch.Tensor:
        """Standardize output y using stored statistics."""
        if self.y_mean is None or self.y_std is None:
            return y
        return (y - self.y_mean) / self.y_std

    def _destandardize_y(self, y: torch.Tensor) -> torch.Tensor:
        """Convert standardized y back to original scale."""
        if self.y_mean is None or self.y_std is None:
            return y
        return y * self.y_std + self.y_mean

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # Case 1: u is 2D (Batch, P) - Single sample per input
        if u.dim() == 2:
            # Reshape output to (Batch, Rank, OutputDim)
            vx = self.varphi(x).reshape(-1, self.r, self.q)
            vu = self.psi(u).reshape(-1, self.r, self.q)
            # Dot product interaction along the rank dimension, scaled by 1/sqrt(r) to keep output variance O(1)
            return (vx * vu).sum(dim=1) * self.rank_scale
    
        # Case 2: u is 3D (Batch, M_samples, P) - Multiple samples per input
        if u.dim() == 3:
            n, m, p = u.shape
            vx = self.varphi(x).reshape(n, self.r, self.q)
            # Process flattened u, then reshape back to (Batch, M_samples, Rank, Q)
            vu = self.psi(u.reshape(n * m, p)).reshape(n, m, self.r, self.q)
            # Broadcasting vx to match m samples: (Batch, 1, Rank, Q) * (Batch, M, Rank, Q)
            # Scale by 1/sqrt(r) to stabilize variance
            y = (vx[:, None, :, :] * vu).sum(dim=2) * self.rank_scale
            return y
    
        raise ValueError("u must have shape (n,p) or (n,m,p).")

    def logdensity(self, xs: torch.Tensor, ys : torch.Tensor, m : int = 30, tilted : bool = False):
        # Check if inputs are tensors (or if training), otherwise delegate to numpy handler
        if(self._istraining or (isinstance(xs, torch.Tensor) and isinstance(ys, torch.Tensor))):
            xs_, ys_ = xs.reshape(-1, self.d), ys.reshape(-1, self.q)
            
            # Standardize inputs and outputs
            xs_ = self._standardize_x(xs_)
            ys_ = self._standardize_y(ys_)

            delta = self.delta if tilted else 1e-15
            # Sample m latent variables per input
            u = self._sample_u(xs_.shape[0], m, device=xs.device)
            # Predict yhat based on x and random u
            yhat = self.forward(xs_, u)
            
            # Calculate residuals between actual y and generated hypotheses
            residuals = (ys_[:, None, :] - yhat)
            eps = self.eps()
            
            # Compute Gaussian log-likelihood components
            zs = residuals / eps
            rs = zs.pow(2).sum(dim=-1) # Squared Mahalanobis-like distance
            
            # Log-pdf of the Gaussian kernel
            exponents = -0.5 * rs - 0.5 * self.q * math.log(2.0 * math.pi) - torch.log(eps).sum() - math.log(m) - math.log(delta)
            
            # Prepare for LogSumExp
            shape = list(exponents.shape)
            shape[1] = 1
            # Compute log mean exp (using stable logsumexp trick) over m samples
            # Includes a stability term (zeros) to prevent -inf
            logd = torch.logsumexp(torch.cat([exponents, torch.zeros(*shape, device = xs.device)], dim=1), dim=1) + math.log(delta)  
            
            # Return scalar if input was single item, else tensor
            return logd if(len(xs.shape)==2 or xs_.shape[0]>1 or len(ys.shape)==2 or ys_.shape[0]>1) else logd[0] 
        else:
            # Handle Numpy inputs by converting to Tensor and recurring
            device = self.eps().device
            return self.logdensity(torch.tensor(xs, device=device, dtype=torch.float32), torch.tensor(ys, device=device, dtype=torch.float32), m = m, tilted = tilted).cpu().numpy()

    def sample_conditional(self, x: torch.Tensor, num_samples: int = 1, seed: Optional[int] = None) -> torch.Tensor:
        if(self._istraining or isinstance(x, torch.Tensor)):
            x_ = x.reshape(-1, self.d)
            x_ = self._standardize_x(x_)
            
            # Setup generator for reproducibility
            g = None
            if seed is not None:
                g = torch.Generator(device=x.device)
                g.manual_seed(int(seed))

            # Sample latent variable u
            if self.latent_dist == "uniform":
                u = torch.rand(x_.shape[0], num_samples, self.p, generator=g, device=x.device)
            else:
                u = torch.randn(x_.shape[0], num_samples, self.p, generator=g, device=x.device)
    
            # Get location parameter (mean of the kernel)
            y_loc = self.forward(x_, u)
            
            # Sample noise from Normal(0,1)
            # We use the same generator 'g' if provided to ensure full reproducibility
            noise = torch.randn(y_loc.shape, generator=g, device=x.device)

            # Add kernel noise (bandwidth) *before* destandardization
            # self.eps() is the learned bandwidth in standardized space
            y = y_loc + self.eps() * noise

            # Destandardize to get back to original data scale
            y = self._destandardize_y(y)

            return y if(len(x.shape)==2 or self.q>1) else y.flatten()
        else:
            device = self.eps().device
            return self.sample_conditional(torch.tensor(x, device=device, dtype=torch.float32), num_samples = num_samples, seed = seed).cpu().numpy()

    def fit(self, xs, ys, epochs=1000, lr=1e-3, m=30, h0=5e-2, val_split=0.0, batch_size=256):
        if isinstance(xs, torch.Tensor) and isinstance(ys, torch.Tensor):
            self._istraining = True
            device = xs.device
    
            # --- Validation Split Logic ---
            if val_split > 0:
                indices = torch.randperm(xs.shape[0])
                v_size = int(xs.shape[0] * val_split)
                train_idx, val_idx = indices[v_size:], indices[:v_size]
                xt, yt = xs[train_idx], ys[train_idx]
                xv, yv = xs[val_idx], ys[val_idx]
            else:
                xt, yt = xs, ys
                xv, yv = None, None
    
            best_val_loss = float('inf')
            best_state = None
    
            with torch.no_grad():
                self.x_mean = xt.mean(dim=0)
                self.x_std = xt.std(dim=0) + 1e-10
                self.y_mean = yt.mean(dim=0)
                self.y_std = yt.std(dim=0) + 1e-10
                eps0 = torch.tensor([h0] * self.q, device=device, dtype=torch.float32)
                try:
                    self._eps_param.copy_(torch.log(eps0))
                except: pass
    
            opt = torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=lr)
            pbar = tqdm(range(epochs), desc="Training CPFN")
            n_train = xt.shape[0]
    
            for epoch in pbar:
                self.train()
    
                # --- Mini-batch loop ---
                perm = torch.randperm(n_train, device=device)
                epoch_loss = 0.0
                n_batches = 0
                for start in range(0, n_train, batch_size):
                    idx = perm[start:start + batch_size]
                    xb, yb = xt[idx], yt[idx]
    
                    opt.zero_grad()
                    loss = -self.logdensity(xb, yb, m, tilted=True).mean()
                    loss.backward()
                    opt.step()
    
                    epoch_loss += loss.item()
                    n_batches += 1
    
                epoch_loss /= n_batches
    
                # --- Validation Check ---
                curr_loss = epoch_loss
                if xv is not None:
                    self.eval()
                    with torch.no_grad():
                        v_loss = -self.logdensity(xv, yv, m, tilted=False).mean().item()
                        if v_loss < best_val_loss:
                            best_val_loss = v_loss
                            best_state = copy.deepcopy(self.state_dict())
                        curr_loss = v_loss
    
                eps_vals = self.eps().detach().cpu().numpy()
                eps_str = ", ".join([f"{e:.2e}" for e in eps_vals])
                pbar.set_postfix({"loss": f"{curr_loss:.4e}", "bw": eps_str})
    
            if best_state is not None:
                self.load_state_dict(best_state)
                print(f"Restored best model with validation loss: {best_val_loss:.4e}")
    
            self._istraining = False
        else:
            device = self.eps().device
            self.fit(
                torch.tensor(xs.astype(np.float32), device=device, dtype=torch.float32),
                torch.tensor(ys.astype(np.float32), device=device, dtype=torch.float32),
                epochs=epochs, lr=lr, m=m, h0=h0, val_split=val_split, batch_size=batch_size
            )
    def freeze(self):
        # Freeze all parameters (prevent gradient updates)
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def unfreeze(self):
        # Unfreeze all parameters (enable gradient updates)
        for p in self.parameters():
            p.requires_grad_(True)
        self.train()

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)

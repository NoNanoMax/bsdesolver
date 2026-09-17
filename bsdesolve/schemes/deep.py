"""Deep BSDE scheme (Han, Jentzen, E 2018).

Core idea:
  Instead of solving the BSDE backward (as in Euler scheme), we:
  1. Parameterize Y_0 as a learnable scalar.
  2. Parameterize Z(t_k, x) as a shared neural network Z_θ(t, x) → R^{dim_w}.
  3. Simulate the BSDE FORWARD:
       X_{k+1} = X_k + μ(t_k, X_k) dt + σ(t_k, X_k) ΔW_k
       Y_{k+1} = Y_k - f(t_k, Y_k, Z_k) dt + Z_k^T ΔW_k
       Z_k = Z_θ(t_k, X_k)
  4. Loss = mean((Y_N - g(X_N))^2)  — terminal mismatch.
  5. Train with Adam + LR schedule.

The solution:
  - Y_0 = the PDE/BSDE value at t=0 (the "price").
  - Z_θ(t, x) = the gradient/hedge function (can be evaluated at any (t, x)).

Convergence: O(dt) time + O(M^{-1/2}) MC + O(δ²) NN approximation.
Dimension-independent (no curse of dimensionality).

References:
  - Han, Jentzen, E (2018). "Solving high-dimensional PDEs using
    deep learning." Proc. NA.
  - Han, Jentzen, E (2017). "A deep learning-based numerical method
    for solving high-dimensional parabolic PDEs."
"""

from __future__ import annotations

import time as _time

import torch
import torch.nn as nn
import torch.nn.functional as F

from bsdesolve.core.problem import BSDEProblem
from bsdesolve.diagnostics.convergence import ConvergenceReport
from bsdesolve.schemes.scheme import BSDEResult, Scheme


class ZNetwork(nn.Module):
    """Shared neural network approximating Z(t, x) ∈ R^{dim_w}.

    The same network is used at all time steps; time t is an input feature.
    This is the "parameter sharing" variant of Deep BSDE (fewer parameters,
    better generalization, especially for high dimensions).

    Args:
        dim_x: Dimension of the state space x.
        dim_w: Dimension of the Brownian motion (output dim of Z).
        hidden_dim: Number of neurons in hidden layers.
        n_layers: Number of hidden layers.
    """

    def __init__(
        self,
        dim_x: int,
        dim_w: int,
        hidden_dim: int = 64,
        n_layers: int = 3,
    ):
        super().__init__()
        input_dim = dim_x + 1  # (t, x_1, ..., x_dim)

        # Build the network
        layers: list[nn.Module] = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(n_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            ])
        layers.append(nn.Linear(hidden_dim, dim_w))
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Evaluate Z_θ(t, x).

        Args:
            t: Scalar time (or tensor of shape (M,)).
            x: State tensor, shape (M, dim_x).

        Returns:
            Z values, shape (M, dim_w).
        """
        M = x.shape[0]
        if t.dim() == 0:
            t_expanded = t.item() * torch.ones(M, 1, device=x.device, dtype=x.dtype)
        else:
            t_expanded = t.reshape(M, 1)
        inp = torch.cat([t_expanded, x], dim=1)  # (M, dim_x + 1)
        return self.net(inp)  # (M, dim_w)

    def extra_repr(self) -> str:
        return f"dim_x={self.net[0].in_features - 1}, hidden={self.net[0].out_features}"


class DeepScheme(Scheme):
    """Deep BSDE solver (Han, Jentzen, E 2018).

    Solves BSDEs by forward simulation + neural network approximation of Z.
    Suitable for high-dimensional problems (10D, 100D, ...).
    """

    def solve(
        self,
        problem: BSDEProblem,
        num_paths: int = 10_000,
        num_time_steps: int = 100,
        device: str = "cpu",
        seed: int | None = None,
        return_diagnostics: bool = False,
        hidden_dim: int = 64,
        n_layers: int = 3,
        learning_rate: float = 1e-3,
        max_iterations: int = 1000,
        batch_size: int | None = None,
        lr_decay: float = 0.95,
        lr_decay_interval: int = 200,
        early_stop_patience: int | None = None,
        verbose: bool = False,
    ) -> BSDEResult:
        """Solve a BSDE using the Deep BSDE method.

        Args:
            problem: The BSDE problem (generator, terminal condition, SDE).
            num_paths: Number of paths (used for training batches).
            num_time_steps: Number of time steps N.
            device: "cpu" or "cuda".
            seed: Random seed.
            return_diagnostics: Whether to compute convergence diagnostics.
            hidden_dim: Hidden layer width.
            n_layers: Number of hidden layers.
            learning_rate: Initial Adam learning rate.
            max_iterations: Maximum number of training iterations.
            batch_size: Mini-batch size (default: num_paths).
            lr_decay: Multiplicative LR decay factor.
            lr_decay_interval: Apply decay every this many iterations.
            early_stop_patience: Stop if loss doesn't improve for this many
                iterations (None = no early stopping).
            verbose: Print training progress.

        Returns:
            BSDEResult with Y, Z, and the trained network in ``.deep`` attribute.
        """
        if seed is not None:
            torch.manual_seed(seed)
            if device == "cuda":
                torch.cuda.manual_seed(seed)

        sde = problem.sde
        if sde is None:
            raise ValueError(
                "Deep BSDE requires a forward SDE (sde must not be None)."
            )

        dim = sde.dim
        dim_w = dim  # standard Markov case

        t0, T = problem.t_span
        N = num_time_steps
        dt = (T - t0) / N
        t_grid = torch.linspace(t0, T, N + 1, device=device)

        if batch_size is None:
            batch_size = num_paths

        # --- Parameterize ---
        y0 = nn.Parameter(torch.zeros(1, device=device))
        z_net = ZNetwork(
            dim_x=dim,
            dim_w=dim_w,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        ).to(device)

        optimizer = torch.optim.Adam(
            [y0, *z_net.parameters()], lr=learning_rate
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: lr_decay ** (step // lr_decay_interval),
        )

        # --- Training loop ---
        best_loss = float("inf")
        best_iter = 0
        patience_counter = 0
        t_start = _time.time()
        losses: list[float] = []

        for it in range(max_iterations):
            # Sample a batch of Brownian increments
            # dW: (batch_size, N, dim_w)
            dW = torch.randn(batch_size, N, dim_w, device=device) * dt**0.5

            # Forward simulation of X and Y
            x = sde.x0
            if isinstance(x, (int, float)):
                x = torch.full((batch_size, dim), float(x), device=device)
            else:
                x = x.to(device).reshape(1, dim).expand(batch_size, dim).clone()

            y = y0.expand(batch_size)  # (batch_size,) — all paths start at y0

            for k in range(N):
                t_k = t_grid[k]
                # Z_k = Z_net(t_k, x)
                z_k = z_net(t_k, x)  # (batch_size, dim_w)
                # X_{k+1} = X_k + mu dt + sigma dW_k
                mu = sde.drift(t_k, x)  # (batch_size, dim)
                sigma = sde.diffusion(t_k, x)  # (batch_size, dim, dim_w)
                dx = mu * dt + torch.einsum("nab,nb->na", sigma, dW[:, k])
                x = x + dx
                # Y_{k+1} = Y_k - f dt + Z_k^T dW_k
                f_val = problem.generator(t_k, y, z_k)  # (batch_size,)
                dy = -f_val * dt + (z_k * dW[:, k]).sum(dim=-1)
                y = y + dy

            # Terminal condition
            g_terminal = problem.terminal_condition(x, t_grid[-1])  # (batch_size,)
            g_terminal = g_terminal.reshape(batch_size)

            # Loss: mean squared terminal mismatch
            loss = F.mse_loss(y, g_terminal)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(
                [y0, *z_net.parameters()], max_norm=10.0
            )
            optimizer.step()
            scheduler.step()

            loss_val = loss.item()
            losses.append(loss_val)

            if loss_val < best_loss:
                best_loss = loss_val
                best_iter = it
                if early_stop_patience is not None:
                    patience_counter = 0
                else:
                    patience_counter = None
            else:
                if early_stop_patience is not None:
                    patience_counter += 1
                    if patience_counter >= early_stop_patience:
                        if verbose:
                            print(f"  Early stopping at iter {it + 1} "
                                  f"(best at {best_iter + 1})")
                        break

            if verbose and (it % 50 == 0 or it == max_iterations - 1):
                elapsed = _time.time() - t_start
                print(f"  iter {it + 1:5d}  loss={loss_val:.6f}  "
                      f"best={best_loss:.6f}  lr={optimizer.param_groups[0]['lr']:.2e}  "
                      f"({elapsed:.1f}s)")

        elapsed_total = _time.time() - t_start
        if verbose:
            print(f"  Training done: {max_iterations} iters, {elapsed_total:.1f}s, "
                  f"final loss={losses[-1]:.6f}")

        # --- Evaluate on a set of test paths to build BSDEResult ---
        # We simulate a fixed number of paths (num_paths) with the trained network
        with torch.no_grad():
            z_net.eval()
            dW_test = torch.randn(num_paths, N, dim_w, device=device) * dt**0.5

            x = sde.x0
            if isinstance(x, (int, float)):
                x = torch.full((num_paths, dim), float(x), device=device)
            else:
                x = x.to(device).reshape(1, dim).expand(num_paths, dim).clone()

            y_val = y0.item()  # scalar
            Y = torch.zeros(num_paths, N + 1, device=device)
            Z = torch.zeros(num_paths, N, dim_w, device=device)
            Y[:, 0] = y_val

            for k in range(N):
                t_k = t_grid[k]
                z_k = z_net(t_k, x)  # (num_paths, dim_w)
                Z[:, k] = z_k
                mu = sde.drift(t_k, x)
                sigma = sde.diffusion(t_k, x)
                dx = mu * dt + torch.einsum("nab,nb->na", sigma, dW_test[:, k])
                x = x + dx
                f_val = problem.generator(t_k, y_val * torch.ones(num_paths, device=device), z_k)
                dy = -f_val * dt + (z_k * dW_test[:, k]).sum(dim=-1)
                y_val_tensor = y_val + dy  # (num_paths,)
                Y[:, k + 1] = y_val_tensor

        z_net.train()  # reset to training mode

        # --- Build result ---
        result = BSDEResult(
            Y=Y,
            Z=Z,
            X=None,  # X is path-dependent; for Deep BSDE the network IS the solution
            t_grid=t_grid,
        )

        # Attach the trained network as an extra attribute
        result.deep = {  # type: ignore[attr-defined]
            "y0": y0,
            "z_net": z_net,
            "losses": losses,
            "training_time": elapsed_total,
            "final_lr": optimizer.param_groups[0]["lr"],
        }

        if return_diagnostics:
            mc_std = Y[:, 0].std().item() / num_paths**0.5
            conv = ConvergenceReport(
                num_paths=num_paths,
                num_time_steps=N,
                dt=dt,
                mc_std_err=mc_std,
                y0_mean=Y[:, 0].mean().item(),
                y0_std=Y[:, 0].std().item(),
            )
            result.convergence = conv

        return result

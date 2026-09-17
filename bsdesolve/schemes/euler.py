"""Backward Euler-Maruyama scheme for BSDE (Monte Carlo + regression).

Algorithm:
  1. Simulate N forward SDE paths (and record the Brownian increments ΔW):
         X_{i+1} = X_i + μ(t_i, X_i) dt + σ(t_i, X_i) ΔW_i
  2. Set Y_T = ξ(X_T) for all paths
  3. For i = T-1, T-2, ..., 0 (backward):
       target_n = Y_{i+1,n} - f(t_i, Y_{i+1,n}, Z_{i+1,n}) * dt
       Regress target on ΔW_i via OLS:
         target ≈ Y_i + Z_i^T ΔW_i
         =>  Z_i = (ΔW^T ΔW)^{-1} ΔW^T (target - mean)   (slope)
         =>  Y_i = mean(target - Z_i^T ΔW_i)             (intercept)
  4. Return Y_0 (the BSDE value at t=0) and Z (the hedge).

This is the classical Monte Carlo approach to solving BSDEs (Bally, Crisan,
Delarue 2008; Han, Jentzen, E 2018 linear case).

Convergence rate: O(dt) in time, O(1/√N) in number of paths.
"""

from __future__ import annotations

from typing import Any

import torch

from bsdesolve.core.problem import BSDEProblem
from bsdesolve.diagnostics.convergence import ConvergenceReport
from bsdesolve.schemes.scheme import BSDEResult, Scheme


class EulerScheme(Scheme):
    """Backward Euler-Maruyama scheme with OLS regression for Z estimation."""

    def solve(
        self,
        problem: BSDEProblem,
        num_paths: int = 10_000,
        num_time_steps: int = 100,
        device: str = "cpu",
        seed: int | None = None,
        return_diagnostics: bool = False,
        **scheme_kwargs: Any,
    ) -> BSDEResult:
        t0, T = problem.t_span
        T_steps = num_time_steps
        dt = (T - t0) / T_steps
        t_grid = torch.linspace(t0, T, T_steps + 1, device=device)

        if seed is not None:
            torch.manual_seed(seed)

        sde = problem.sde
        if sde is not None:
            dim = sde.dim
            # Brownian dimension = state dimension (standard Markov case)
            dim_w = dim
        else:
            dim = 1
            dim_w = 1

        # --- Step 1: Simulate forward SDE, recording ΔW ---
        # ΔW_i ~ N(0, dt * I) for each step
        dW = torch.randn(num_paths, T_steps, dim_w, device=device) * dt**0.5
        # dW[:, i] is the increment over [t_i, t_{i+1}]

        if sde is not None:
            X = torch.empty(num_paths, T_steps + 1, dim, device=device)
            x0 = sde.x0
            if isinstance(x0, (int, float)):
                x0 = torch.full((dim,), float(x0), device=device)
            else:
                x0 = x0.to(device=device).reshape(dim)
            X[:, 0] = x0

            for i in range(T_steps):
                t_i = t_grid[i]
                X_i = X[:, i]  # (N, dim)
                mu = sde.drift(t_i, X_i)  # (N, dim)
                sigma = sde.diffusion(t_i, X_i)  # (N, dim, dim_w)
                # dx = μ dt + σ ΔW
                dX = mu * dt + torch.einsum("nab,nb->na", sigma, dW[:, i])
                X[:, i + 1] = X_i + dX
            X_paths = X
        else:
            # Pure Brownian motion (no driving SDE)
            W = torch.zeros(num_paths, T_steps + 1, dim_w, device=device)
            W[:, 1:] = dW.cumsum(dim=1)
            X_paths = W.unsqueeze(-1)  # (N, T+1, 1)

        # --- Step 2: Terminal condition ---
        X_T = X_paths[:, -1] if sde is not None else X_paths[:, -1, 0]
        Y_T = problem.terminal_condition(X_T, t_grid[-1])  # (N,)
        Y_T = Y_T.reshape(num_paths)

        # --- Step 3: Backward Euler-Maruyama ---
        Y = torch.zeros(num_paths, T_steps + 1, device=device)
        Z = torch.zeros(num_paths, T_steps, dim_w, device=device)
        Y[:, -1] = Y_T

        step_errors: list[float] = []
        y_means: list[float] = []
        y_vars: list[float] = []

        # Precompute mean of dW for centered regression (optional but faster)
        for i in range(T_steps - 1, -1, -1):
            # Y_{i+1} already known; Z_{i+1} known (Z[:, i+1]) if i+1 < T
            Y_next = Y[:, i + 1]  # (N,)
            Z_next = Z[:, i + 1] if i + 1 < T_steps else torch.zeros(
                num_paths, dim_w, device=device
            )  # (N, dim_w)

            # Generator f(t, Y, Z) evaluated at t_i (current step)
            t_val = t_grid[i]
            f_val = problem.generator(t_val, Y_next, Z_next)  # (N,)
            f_val = f_val.reshape(num_paths)

            # Target for this step
            target = Y_next - f_val * dt  # (N,)

            # OLS regression of target on ΔW_i:
            #   target ≈ Y_i + Z_i^T ΔW_i
            dW_i = dW[:, i]  # (N, dim_w)
            Z_i, Y_i = _ols_regression(target, dW_i)

            Y[:, i] = Y_i
            Z[:, i] = Z_i

            if return_diagnostics:
                step_errors.append(target.std().item())
                y_means.append(Y[:, i].mean().item())
                y_vars.append(Y[:, i].var().item())

        # --- Step 4: Build result ---
        result = BSDEResult(
            Y=Y,
            Z=Z,
            X=X_paths if sde is not None else None,
            t_grid=t_grid,
        )

        if return_diagnostics:
            mc_std = Y[:, 0].std().item() / num_paths**0.5
            conv = ConvergenceReport(
                num_paths=num_paths,
                num_time_steps=T_steps,
                dt=dt,
                mc_std_err=mc_std,
                y0_mean=Y[:, 0].mean().item(),
                y0_std=Y[:, 0].std().item(),
                step_errors=step_errors,
                y_means=y_means,
                y_vars=y_vars,
            )
            result.convergence = conv

        return result


def _ols_regression(target: torch.Tensor, dW: torch.Tensor) -> tuple[torch.Tensor, float]:
    """OLS regression: target ≈ Y + Z^T dW.

    Returns (Z, Y) where Z has shape (dim_w,) and Y is a scalar (float).

    Uses the centered form:
        Z = (ΔW^T ΔW)^{-1} ΔW^T (target - mean_target)
        Y = mean(target) - Z^T mean(ΔW)
    """
    N = target.shape[0]
    dim_w = dW.shape[1]

    target_c = target - target.mean()
    dW_c = dW - dW.mean(dim=0)  # (N, dim_w)

    # Z = (dW_c^T dW_c)^{-1} dW_c^T target_c
    #   = (ΔW^T ΔW / N)^{-1} (ΔW^T target_c / N)
    A = (dW_c.T @ dW_c) / N  # (dim_w, dim_w)
    b = (dW_c.T @ target_c) / N  # (dim_w,)

    # Solve A @ Z = b (with regularization for stability)
    reg = 1e-8 * torch.eye(dim_w, device=A.device, dtype=A.dtype)
    try:
        Z = torch.linalg.solve(A + reg, b)
    except torch.linalg.LinAlgError:
        # Fallback: pseudoinverse
        Z = torch.linalg.pinv(A) @ b

    Y = target.mean().item() - float(Z @ dW.mean(dim=0))

    return Z, Y

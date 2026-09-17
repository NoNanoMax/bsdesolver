"""Abstract base class for BSDE numerical schemes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

from bsdesolve.core.problem import BSDEProblem
from bsdesolve.diagnostics.convergence import ConvergenceReport


@dataclass
class BSDEResult:
    """Result of solving a BSDE.

    Attributes:
        Y: Value process (backward component), shape (num_paths, T+1).
            Y[:, i] is Y at time t_grid[i].
        Z: Diffusion/hedge process, shape (num_paths, T, dim_w).
            Z[:, i] is Z at time interval [t_i, t_{i+1}].
        X: Forward SDE paths (if coupled), shape (num_paths, T+1, dim).
            None for pure BSDE (no forward SDE).
        convergence: Convergence diagnostics (if requested).
        t_grid: Time grid, shape (T+1,).
    """

    Y: torch.Tensor
    Z: torch.Tensor
    X: torch.Tensor | None = None
    convergence: ConvergenceReport | None = None
    t_grid: torch.Tensor | None = None

    @property
    def Y0(self) -> torch.Tensor:
        """Value at t=0, shape (num_paths,)."""
        return self.Y[:, 0]

    @property
    def Y0_mean(self) -> float:
        """Mean value at t=0 (Monte Carlo estimate)."""
        return self.Y[:, 0].mean().item()

    def __repr__(self) -> str:
        return (
            f"BSDEResult(Y.shape={self.Y.shape}, Z.shape={self.Z.shape}, "
            f"Y0_mean={self.Y0_mean:.6f})"
        )


class Scheme(ABC):
    """Abstract base class for BSDE numerical schemes."""

    @abstractmethod
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
        """Solve the BSDE.

        Args:
            problem: The BSDE problem definition.
            num_paths: Number of Monte Carlo paths.
            num_time_steps: Number of time steps (T = num_time_steps).
            device: "cpu" or "cuda".
            seed: Random seed.
            return_diagnostics: Whether to compute convergence diagnostics.
            **scheme_kwargs: Scheme-specific parameters (e.g. for "deep":
                hidden_dim, n_layers, learning_rate, max_iterations, ...).

        Returns:
            BSDEResult with Y, Z, and optional diagnostics.
        """
        ...

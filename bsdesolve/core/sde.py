"""Forward SDE definitions (driving the filtration for BSDE/FBSDE)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class ForwardSDE(ABC):
    """Abstract base class for forward SDEs.

    The forward SDE drives the filtration for the BSDE:
        dX_t = μ(t, X_t) dt + σ(t, X_t) dW_t,   X_0 = x0

    where:
        μ: drift
        σ: diffusion (volatility)
        W: d-dimensional Brownian motion
        x0: initial condition

    The dimension of X determines the dimension of W (in the standard case).
    """

    def __init__(self, dim: int = 1, x0: float | torch.Tensor = 1.0):
        self.dim = dim
        self.x0 = x0

    @abstractmethod
    def drift(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Drift coefficient μ(t, x).

        Args:
            t: Time, shape (N,) or scalar.
            x: State, shape (N, dim).

        Returns:
            Drift, shape (N, dim).
        """
        ...

    @abstractmethod
    def diffusion(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Diffusion coefficient σ(t, x).

        Args:
            t: Time, shape (N,) or scalar.
            x: State, shape (N, dim).

        Returns:
            Diffusion, shape (N, dim, dim_w) where dim_w is the number
            of Brownian motions.
        """
        ...

    def simulate(
        self,
        t_grid: torch.Tensor,
        num_paths: int,
        device: str = "cpu",
        seed: int | None = None,
    ) -> torch.Tensor:
        """Simulate the SDE using Euler-Maruyama.

        Args:
            t_grid: Time grid, shape (T+1,).
            num_paths: Number of paths to simulate.
            device: Device ("cpu" or "cuda").
            seed: Random seed for reproducibility.

        Returns:
            Paths tensor, shape (num_paths, T+1, dim).
        """
        if seed is not None:
            torch.manual_seed(seed)

        dt = (t_grid[1] - t_grid[0]).item()
        dW = torch.randn(num_paths, len(t_grid) - 1, self.dim, device=device) * dt**0.5

        paths = torch.empty(num_paths, len(t_grid), self.dim, device=device)
        x0 = self.x0
        if isinstance(x0, (int, float)):
            x0 = torch.full((self.dim,), x0, device=device)
        else:
            x0 = x0.to(device=device)
        paths[:, 0] = x0

        for i in range(1, len(t_grid)):
            t = t_grid[i - 1]
            x = paths[:, i - 1]  # (N, dim)
            mu = self.drift(t, x)  # (N, dim)
            sigma = self.diffusion(t, x)  # (N, dim, dim_w)
            # sigma @ dW: (N, dim, dim_w) @ (N, dim_w) -> (N, dim)
            dx = mu * dt + torch.einsum("nab,nb->na", sigma, dW[:, i - 1])
            paths[:, i] = x + dx

        return paths

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dim={self.dim}, x0={self.x0})"


class GBM(ForwardSDE):
    """Geometric Brownian Motion: dS_t = μ S_t dt + σ S_t dW_t.

    The canonical model for stock prices.

    Args:
        drift: Drift μ (risk-neutral or physical).
        diffusion: Volatility σ.
        x0: Initial price S_0.
    """

    def __init__(self, drift: float = 0.05, diffusion: float = 0.2, x0: float = 100.0):
        super().__init__(dim=1, x0=x0)
        self.mu = drift
        self.sigma = diffusion

    def drift(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # x is (N, 1), return (N, 1)
        return self.mu * x

    def diffusion(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # x is (N, 1), return (N, 1, 1)
        return self.sigma * x.unsqueeze(-1)

    def __repr__(self) -> str:
        return f"GBM(μ={self.mu}, σ={self.sigma}, S₀={self.x0})"


class OU(ForwardSDE):
    """Ornstein-Uhlenbeck: dX_t = θ(μ - X_t) dt + σ dW_t.

    Mean-reverting process, widely used for interest rates, volatility.

    Args:
        theta: Mean reversion speed θ.
        mu: Long-term mean μ.
        sigma: Volatility σ.
        x0: Initial value.
    """

    def __init__(
        self,
        theta: float = 2.0,
        mu: float = 0.0,
        sigma: float = 0.1,
        x0: float = 1.0,
    ):
        super().__init__(dim=1, x0=x0)
        self.theta = theta
        self.mu = mu
        self.sigma = sigma

    def drift(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.theta * (self.mu - x)

    def diffusion(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        return torch.full((N, 1, 1), self.sigma, dtype=x.dtype, device=x.device)

    def __repr__(self) -> str:
        return f"OU(θ={self.theta}, μ={self.mu}, σ={self.sigma})"


class CIR(ForwardSDE):
    """Cox-Ingersoll-Ross: dX_t = θ(μ - X_t) dt + σ √X_t dW_t.

    Square-root diffusion, used for interest rates and variance.

    Args:
        theta: Mean reversion speed θ.
        mu: Long-term mean μ.
        sigma: Volatility σ.
        x0: Initial value.
    """

    def __init__(
        self,
        theta: float = 2.0,
        mu: float = 0.05,
        sigma: float = 0.1,
        x0: float = 0.05,
    ):
        super().__init__(dim=1, x0=x0)
        self.theta = theta
        self.mu = mu
        self.sigma = sigma

    def drift(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.theta * (self.mu - x)

    def diffusion(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # σ * √x, shape (N, 1) -> (N, 1, 1)
        return self.sigma * torch.sqrt(x.clamp(min=0.0)).unsqueeze(-1)

    def __repr__(self) -> str:
        return f"CIR(θ={self.theta}, μ={self.mu}, σ={self.sigma})"


class BlackScholes(GBM):
    """Black-Scholes GBM with risk-neutral drift = r.

    Convenience subclass for pricing.

    Args:
        r: Risk-free rate.
        sigma: Volatility.
        x0: Initial spot price.
    """

    def __init__(self, r: float = 0.05, sigma: float = 0.2, x0: float = 100.0):
        super().__init__(drift=r, diffusion=sigma, x0=x0)
        self.r = r
        self.sigma_ = sigma

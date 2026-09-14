"""Terminal condition definitions for BSDE."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch


class TerminalCondition(ABC):
    """Abstract base class for BSDE terminal conditions.

    The terminal condition ξ defines Y_T.
    """

    @abstractmethod
    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the terminal condition.

        Args:
            x: Terminal state of the forward SDE, shape (N,) or (N, dim).
            t: Terminal time, shape (N,) or scalar.

        Returns:
            Terminal condition value, shape (N,) or (N, dim).
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ConstantTerminal(TerminalCondition):
    """Constant terminal condition: ξ = c.

    Args:
        value: Constant value (scalar).
    """

    def __init__(self, value: float = 0.0):
        self.value = value

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        return torch.full((N,), self.value, dtype=x.dtype, device=x.device)

    def __repr__(self) -> str:
        return f"ConstantTerminal(value={self.value})"


class FunctionTerminal(TerminalCondition):
    """Function terminal condition: ξ = g(x_T).

    For a forward SDE dX_t = μ dt + σ dW, the terminal condition is a
    function of the terminal state: ξ = g(X_T).

    Example (European call): g(x) = max(x - K, 0)

    Args:
        func: A callable g(x) -> tensor.
    """

    def __init__(self, func: Callable[[torch.Tensor], torch.Tensor]):
        self.func = func

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.func(x)

    def __repr__(self) -> str:
        return f"FunctionTerminal(func={self.func.__name__})"


class PathDependentTerminal(TerminalCondition):
    """Path-dependent terminal condition: ξ = g(x_T, path).

    For path-dependent payoffs (barrier, lookback, Asian), the terminal
    condition depends on the full path.

    Args:
        func: A callable g(x_T, path) -> tensor, where path has shape (N, T, dim).
    """

    def __init__(self, func: Callable[..., torch.Tensor]):
        self.func = func

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x is (N,) or (N, dim), t is scalar or (N,)
        # For path-dependent, we need the full path — this is a placeholder
        # that will be extended when we implement path-dependent problems.
        # For now, treat x as the full state (last time step).
        return self.func(x)

    def __repr__(self) -> str:
        return f"PathDependentTerminal(func={self.func.__name__})"

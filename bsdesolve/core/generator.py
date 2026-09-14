"""Generator (drift) definitions for BSDE."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch

# A generator coefficient: constant, time-dependent callable, or precomputed tensor.
Coeff = float | Callable[[torch.Tensor], torch.Tensor] | torch.Tensor


class Generator(ABC):
    """Abstract base class for BSDE generators.

    The generator f(t, y, z) defines the drift of the BSDE:
        dY_t = -f(t, Y_t, Z_t) dt + Z_t dW_t
    """

    @abstractmethod
    def __call__(self, t: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Evaluate the generator.

        Args:
            t: Time tensor, shape (N,) or scalar.
            y: Value process, shape (N,) or (N, dim).
            z: Diffusion coefficient, shape (N,) or (N, dim_w).

        Returns:
            Generator value, shape (N,) or (N, dim).
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class LinearGenerator(Generator):
    """Linear generator: f(t, y, z) = a(t) * y + b(t) * z + c(t).

    This is the most common linear case, e.g. risk-free rate:
        f(t, y, z) = r(t) * y  (i.e. a = r, b = 0, c = 0)

    Args:
        a: Coefficient of y. Can be float, callable(t), or tensor.
        b: Coefficient of z. Can be float, callable(t), or tensor.
        c: Additive term. Can be float, callable(t), or tensor.
    """

    def __init__(
        self,
        a: Coeff = 0.0,
        b: Coeff = 0.0,
        c: Coeff = 0.0,
    ):
        self.a = a
        self.b = b
        self.c = c

    def _eval(self, val: Coeff, t: torch.Tensor) -> torch.Tensor:
        """Evaluate a coefficient that can be scalar, callable, or tensor."""
        if callable(val):
            return val(t)
        if isinstance(val, torch.Tensor):
            return val
        return torch.full_like(t, float(val), dtype=t.dtype)

    def __call__(self, t: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Evaluate f(t, y, z) = a(t)*y + b(t)*z + c(t).

        The output is always shape (N,) — one scalar per path.
        If z has shape (N, dim_w), it is reduced to (N,) by summing
        over the last dimension (dot product with ones).
        """
        a = self._eval(self.a, t)
        b = self._eval(self.b, t)
        c = self._eval(self.c, t)

        # y is (N,)
        y = y.reshape(-1)
        # z is (N, dim_w) or (N,) — reduce to (N,)
        z_reduced = z.sum(dim=-1) if z.dim() > 1 else z.reshape(-1)

        # a, b, c are scalar or (N,)
        result = a * y + b * z_reduced + c
        return result.reshape(-1)

    def __repr__(self) -> str:
        return f"LinearGenerator(a={self.a}, b={self.b}, c={self.c})"


class NonlinearGenerator(Generator):
    """User-defined nonlinear generator: f(t, y, z) = f_custom(t, y, z).

    Args:
        func: A callable f(t, y, z) -> tensor.
    """

    def __init__(self, func: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]):
        self.func = func

    def __call__(self, t: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.func(t, y, z)

    def __repr__(self) -> str:
        return f"NonlinearGenerator(func={self.func.__name__})"

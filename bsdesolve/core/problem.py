"""Core problem definitions for BSDE and FBSDE."""

from __future__ import annotations

from dataclasses import dataclass

from bsdesolve.core.generator import Generator
from bsdesolve.core.sde import ForwardSDE
from bsdesolve.core.terminal import TerminalCondition


@dataclass
class BSDEProblem:
    """Defines a Backward Stochastic Differential Equation (BSDE).

    The BSDE is:
        dY_t = -f(t, Y_t, Z_t) dt + Z_t dW_t,   Y_T = ξ

    where:
        f: generator (drift) — can be nonlinear in (y, z)
        ξ: terminal condition — random variable at time T
        W: Brownian motion (dimension determined by SDE)

    For a coupled FBSDE, the forward SDE drives the filtration:
        dX_t = μ(t, X_t) dt + σ(t, X_t) dW_t

    Args:
        generator: The BSDE generator f(t, y, z).
        terminal_condition: The terminal condition ξ.
        sde: The forward SDE driving the filtration (if any).
        t_span: Time interval (t0, T).
        dim: Dimension of the state space (for SDE).
    """

    generator: Generator
    terminal_condition: TerminalCondition
    sde: ForwardSDE | None = None
    t_span: tuple[float, float] = (0.0, 1.0)
    dim: int = 1
    name: str = "bsde"


@dataclass
class FBSDEProblem:
    """Defines a Forward-Backward Stochastic Differential Equation (FBSDE).

    The FBSDE system is:
        dX_t = μ(t, X_t) dt + σ(t, X_t) dW_t,          X_0 = x0
        dY_t = -f(t, X_t, Y_t, Z_t) dt + Z_t dW_t,     Y_T = ξ(X_T)

    For strongly coupled systems, f may depend on (Y, Z) in a way that
    requires Picard/fixed-point iteration.

    Args:
        generator: The BSDE generator f(t, x, y, z).
        terminal_condition: The terminal condition ξ.
        sde: The forward SDE.
        t_span: Time interval (t0, T).
        dim: Dimension of the state space.
        coupling: Type of coupling: "weak" (f depends on X) or "strong" (f depends on X, Y, Z).
        name: Problem name.
    """

    generator: Generator
    terminal_condition: TerminalCondition
    sde: ForwardSDE
    t_span: tuple[float, float] = (0.0, 1.0)
    dim: int = 1
    coupling: str = "weak"  # "weak" | "strong"
    name: str = "fbsde"

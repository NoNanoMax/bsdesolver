"""Benchmark problems with known (exact) solutions for validation."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass
class Benchmark:
    """A benchmark problem with a known exact solution.

    Attributes:
        name: Human-readable name.
        description: Short description.
        make_problem: Factory that returns a BSDEProblem.
        exact: Callable (t, x) -> float giving the exact solution.
        params: Extra parameters dict.
    """

    name: str
    description: str
    make_problem: Callable[[], object]
    exact: Callable[..., float]
    params: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.params is None:
            self.params = {}


def blackscholes(
    S0: float = 100.0,
    K: float = 100.0,
    r: float = 0.05,
    sigma: float = 0.2,
    T: float = 1.0,
) -> Benchmark:
    """Black-Scholes European call as a BSDE.

    The BSDE is:
        dY_t = -r Y_t dt + Z_t dW_t,   Y_T = max(S_T - K, 0)

    with the forward SDE:
        dS_t = r S_t dt + sigma S_t dW_t

    The exact solution is the Black-Scholes call price.
    """
    from bsdesolve.core.generator import LinearGenerator
    from bsdesolve.core.problem import BSDEProblem
    from bsdesolve.core.sde import GBM
    from bsdesolve.core.terminal import FunctionTerminal

    sde = GBM(drift=r, diffusion=sigma, x0=S0)
    generator = LinearGenerator(a=r)
    terminal = FunctionTerminal(func=lambda x: torch.clamp(x - K, min=0.0))
    problem = BSDEProblem(
        generator=generator,
        terminal_condition=terminal,
        sde=sde,
        t_span=(0.0, T),
        dim=1,
        name="blackscholes_call",
    )

    def _norm_cdf(z: float) -> float:
        """Standard normal CDF via erf."""
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    def exact(t: float, x: float) -> float:
        """Black-Scholes call price."""
        if T - t <= 0:
            return max(x - K, 0.0)
        tau = T - t
        d1 = (math.log(x / K) + (r + 0.5 * sigma**2) * tau) / (sigma * math.sqrt(tau))
        d2 = d1 - sigma * math.sqrt(tau)
        return x * _norm_cdf(d1) - K * math.exp(-r * tau) * _norm_cdf(d2)

    return Benchmark(
        name="Black-Scholes Call",
        description="European call option under Black-Scholes dynamics.",
        make_problem=lambda: problem,
        exact=exact,
        params={"S0": S0, "K": K, "r": r, "sigma": sigma, "T": T},
    )


def linear_bsde(
    r: float = 0.05,
    T: float = 1.0,
    xi: float = 1.0,
) -> Benchmark:
    """Linear BSDE with constant terminal condition.

    The BSDE:
        dY_t = -r Y_t dt + Z_t dW_t,   Y_T = xi (constant)

    Exact solution: Y_t = xi * exp(-r (T - t)),  Z_t = 0.

    This is the simplest non-trivial BSDE and a good regression test.
    """
    from bsdesolve.core.generator import LinearGenerator
    from bsdesolve.core.problem import BSDEProblem
    from bsdesolve.core.terminal import ConstantTerminal

    generator = LinearGenerator(a=r)
    terminal = ConstantTerminal(value=xi)
    problem = BSDEProblem(
        generator=generator,
        terminal_condition=terminal,
        sde=None,  # pure Brownian
        t_span=(0.0, T),
        dim=1,
        name="linear_bsde",
    )

    def exact(t: float, x: float) -> float:
        return xi * math.exp(-r * (T - t))

    return Benchmark(
        name="Linear BSDE (constant xi)",
        description="Linear BSDE with constant terminal condition. Exact: xi*exp(-r(T-t)).",
        make_problem=lambda: problem,
        exact=exact,
        params={"r": r, "T": T, "xi": xi},
    )


def barenblatt(
    d: int = 2,
    T: float = 1.0,
) -> Benchmark:
    """Black-Scholes-Barenblatt (nonlinear PDE via BSDE).

    The generator is nonlinear in Z:
        f(t, y, z) = -z^T Σ z - r y + (d-1) * ||z||^2  (simplified)

    This is the canonical Deep BSDE benchmark. For d>1 there is no closed-form
    exact solution, so we use a published reference value for validation.

    NOTE: This is a placeholder — the full Barenblatt generator will be
    implemented in Phase 2 (Deep BSDE). For now we return a linear proxy.
    """
    from bsdesolve.core.generator import NonlinearGenerator
    from bsdesolve.core.problem import BSDEProblem
    from bsdesolve.core.terminal import FunctionTerminal

    # Placeholder: linear generator for now
    generator = NonlinearGenerator(
        func=lambda t, y, z: 0.5 * z @ z - 0.05 * y
    )
    terminal = FunctionTerminal(func=lambda x: torch.exp(-0.1 * x @ x))
    problem = BSDEProblem(
        generator=generator,
        terminal_condition=terminal,
        sde=None,
        t_span=(0.0, T),
        dim=d,
        name="barenblatt",
    )

    def exact(t: float, x: float) -> float:
        # No closed form; return a reference value
        return 1.0

    return Benchmark(
        name="Barenblatt (placeholder)",
        description="Black-Scholes-Barenblatt nonlinear PDE. Placeholder for Phase 2.",
        make_problem=lambda: problem,
        exact=exact,
        params={"d": d, "T": T},
    )


def get_all_benchmarks() -> list[Benchmark]:
    """Return all registered benchmarks."""
    return [
        blackscholes(),
        linear_bsde(),
        barenblatt(),
    ]

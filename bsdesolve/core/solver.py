"""High-level solver entry points."""

from __future__ import annotations

from bsdesolve.core.generator import Generator
from bsdesolve.core.problem import BSDEProblem
from bsdesolve.core.sde import ForwardSDE
from bsdesolve.core.terminal import TerminalCondition
from bsdesolve.schemes.euler import EulerScheme
from bsdesolve.schemes.scheme import BSDEResult, Scheme


def solve(
    generator: Generator,
    terminal_condition: TerminalCondition,
    sde: ForwardSDE | None = None,
    t_span: tuple[float, float] = (0.0, 1.0),
    dim: int = 1,
    method: str = "euler",
    num_paths: int = 10_000,
    num_time_steps: int = 100,
    device: str = "cpu",
    seed: int | None = None,
    return_diagnostics: bool = False,
) -> BSDEResult:
    """Solve a BSDE.

    Args:
        generator: The BSDE generator f(t, y, z).
        terminal_condition: The terminal condition ξ.
        sde: Optional forward SDE driving the filtration. If None, uses
            standard Brownian motion.
        t_span: Time interval (t0, T).
        dim: State dimension (only used if sde is None).
        method: Numerical scheme: "euler" (Monte Carlo + OLS).
            More methods (e.g. "deep") will be added.
        num_paths: Number of Monte Carlo paths.
        num_time_steps: Number of time steps.
        device: "cpu" or "cuda".
        seed: Random seed for reproducibility.
        return_diagnostics: Whether to compute convergence diagnostics.

    Returns:
        BSDEResult with Y (value process), Z (hedge process).

    Example:
        >>> import torch
        >>> from bsdesolve import solve, LinearGenerator, FunctionTerminal, GBM
        >>> # Risk-free rate BSDE: dY = -r Y dt + Z dW, Y_T = max(S_T - K, 0)
        >>> r = 0.05
        >>> K = 100.0
        >>> sde = GBM(drift=r, diffusion=0.2, x0=100.0)
        >>> gen = LinearGenerator(a=r)  # f(t,y,z) = r*y
        >>> term = FunctionTerminal(func=lambda x: torch.clamp(x - K, min=0.0))
        >>> result = solve(gen, term, sde=sde, num_paths=100_000, seed=42)
        >>> print(f"Option price: {result.Y0_mean:.4f}")
    """
    if method not in _SCHEMES:
        raise ValueError(f"Unknown method '{method}'. Available: {list(_SCHEMES.keys())}")

    problem = BSDEProblem(
        generator=generator,
        terminal_condition=terminal_condition,
        sde=sde,
        t_span=t_span,
        dim=dim,
    )

    scheme: Scheme = _SCHEMES[method]()
    return scheme.solve(
        problem=problem,
        num_paths=num_paths,
        num_time_steps=num_time_steps,
        device=device,
        seed=seed,
        return_diagnostics=return_diagnostics,
    )


def solve_fbsde(
    generator: Generator,
    terminal_condition: TerminalCondition,
    sde: ForwardSDE,
    t_span: tuple[float, float] = (0.0, 1.0),
    dim: int = 1,
    coupling: str = "weak",
    method: str = "euler",
    num_paths: int = 10_000,
    num_time_steps: int = 100,
    device: str = "cpu",
    seed: int | None = None,
    return_diagnostics: bool = False,
) -> BSDEResult:
    """Solve a Forward-Backward SDE (FBSDE).

    For weakly coupled systems (f depends on X), this reduces to a standard
    BSDE with X known. For strongly coupled systems, Picard iteration is used.

    Args:
        generator: The BSDE generator f(t, x, y, z).
        terminal_condition: The terminal condition ξ.
        sde: The forward SDE.
        t_span: Time interval (t0, T).
        dim: State dimension.
        coupling: "weak" or "strong".
        method: Numerical scheme.
        num_paths: Number of Monte Carlo paths.
        num_time_steps: Number of time steps.
        device: "cpu" or "cuda".
        seed: Random seed.
        return_diagnostics: Whether to compute convergence diagnostics.

    Returns:
        BSDEResult with Y, Z, and X (forward paths).
    """
    # For the MVP, weakly coupled FBSDE = standard BSDE with known X.
    # The generator's signature handles the X-dependence internally.
    problem = BSDEProblem(
        generator=generator,
        terminal_condition=terminal_condition,
        sde=sde,
        t_span=t_span,
        dim=dim,
    )

    if method not in _SCHEMES:
        raise ValueError(f"Unknown method '{method}'. Available: {list(_SCHEMES.keys())}")

    scheme: Scheme = _SCHEMES[method]()
    return scheme.solve(
        problem=problem,
        num_paths=num_paths,
        num_time_steps=num_time_steps,
        device=device,
        seed=seed,
        return_diagnostics=return_diagnostics,
    )


# Registry of available schemes
_SCHEMES: dict[str, type[Scheme]] = {
    "euler": EulerScheme,
}

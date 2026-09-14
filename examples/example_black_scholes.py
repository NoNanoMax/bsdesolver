"""Runnable example: price a European call via a BSDE (Euler-Maruyama).

Run:
    python examples/example_black_scholes.py
"""

from __future__ import annotations

import math

import torch

from bsdesolve import (
    GBM,
    FunctionTerminal,
    LinearGenerator,
    solve,
)


def bs_call_exact(S: float, K: float, r: float, sigma: float, tau: float) -> float:
    """Black-Scholes European call price (for comparison)."""
    if tau <= 0:
        return max(S - K, 0.0)

    def phi(z: float) -> float:
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * math.sqrt(tau))
    d2 = d1 - sigma * math.sqrt(tau)
    return S * phi(d1) - K * math.exp(-r * tau) * phi(d2)


def main() -> None:
    # --- Parameters ---
    S0 = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0

    # --- Build the BSDE ---
    #   dY_t = -r Y_t dt + Z_t dW_t,   Y_T = max(S_T - K, 0)
    #   dS_t = r S_t dt + sigma S_t dW_t
    sde = GBM(drift=r, diffusion=sigma, x0=S0)
    generator = LinearGenerator(a=r)
    terminal = FunctionTerminal(func=lambda x: torch.clamp(x - K, min=0.0))

    # --- Solve ---
    result = solve(
        generator=generator,
        terminal_condition=terminal,
        sde=sde,
        t_span=(0.0, T),
        num_paths=100_000,
        num_time_steps=200,
        seed=42,
        return_diagnostics=True,
    )

    exact = bs_call_exact(S0, K, r, sigma, T)

    print("=" * 50)
    print("Black-Scholes European Call via BSDE")
    print("=" * 50)
    print(f"  Parameters : S0={S0}, K={K}, r={r}, sigma={sigma}, T={T}")
    print(f"  MC price   : {result.Y0_mean:.4f}")
    print(f"  Exact (BS) : {exact:.4f}")
    print(f"  Abs error  : {abs(result.Y0_mean - exact):.4f}")
    print()
    if result.convergence is not None:
        print(result.convergence.summary())


if __name__ == "__main__":
    main()

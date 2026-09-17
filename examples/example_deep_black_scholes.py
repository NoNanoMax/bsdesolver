"""Example: Deep BSDE for Black-Scholes option pricing.

Demonstrates the Deep BSDE method (Han, Jentzen, E 2018) for pricing
a European call option under GBM dynamics.

Usage:
    python examples/example_deep_black_scholes.py

On GPU this converges to ~10.45 (exact) in ~2 min with 20k iters.
On CPU, use reduced settings for a quick demonstration.
"""

from __future__ import annotations

import math
import time

import torch

from bsdesolve import (
    GBM,
    FunctionTerminal,
    LinearGenerator,
    solve,
)


def bs_call(S: float, K: float, r: float, sigma: float, tau: float) -> float:
    """Analytic Black-Scholes call price."""
    if tau <= 0:
        return max(S - K, 0.0)
    def phi(z: float) -> float:
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * math.sqrt(tau))
    d2 = d1 - sigma * math.sqrt(tau)
    return S * phi(d1) - K * math.exp(-r * tau) * phi(d2)


def main() -> None:
    # Parameters
    S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0

    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Black-Scholes: S0={S0}, K={K}, r={r}, sigma={sigma}, T={T}")

    exact = bs_call(S0, K, r, sigma, T)
    print(f"Exact call price: {exact:.4f}")
    print()

    # Hyperparameters
    # On GPU: 20000 iters, 8192 paths → high accuracy
    # On CPU: 2000 iters, 1024 paths → rough approximation
    if device == "cuda":
        max_iterations = 20_000
        num_paths = 8192
    else:
        max_iterations = 2000
        num_paths = 1024

    print(f"Settings: {max_iterations} iters, {num_paths} paths")
    print("Training...")

    t0 = time.time()
    result = solve(
        generator=LinearGenerator(a=r),
        terminal_condition=FunctionTerminal(
            func=lambda x: torch.clamp(x - K, min=0.0)
        ),
        sde=GBM(drift=r, diffusion=sigma, x0=S0),
        t_span=(0.0, T),
        method="deep",
        num_paths=num_paths,
        num_time_steps=30,
        device=device,
        seed=42,
        max_iterations=max_iterations,
        hidden_dim=128,
        n_layers=3,
        learning_rate=1e-2,
        lr_decay=0.9,
        lr_decay_interval=2000,
        verbose=True,
    )
    elapsed = time.time() - t0

    # Results
    y0 = result.deep["y0"].item()
    print()
    print("=" * 50)
    print(f"Deep BSDE result:  {y0:.4f}")
    print(f"Exact:             {exact:.4f}")
    print(f"Absolute error:    {abs(y0 - exact):.4f}")
    print(f"Relative error:    {abs(y0 - exact) / exact * 100:.2f}%")
    print(f"Final loss:        {result.deep['losses'][-1]:.6f}")
    print(f"Training time:     {elapsed:.1f}s")
    print("=" * 50)

    # Evaluate Z (delta) at t=0, x=S0
    net = result.deep["z_net"]
    net.eval()
    with torch.no_grad():
        z_at_0 = net(torch.tensor(0.0, device=device),
                     torch.tensor([[S0]], device=device)).item()
    # Analytic delta for ATM call ≈ 0.638
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    delta_exact = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
    print()
    print(f"Z(0, S0) [≈ delta·σ·S]: {z_at_0:.4f}")
    print(f"Analytic delta:         {delta_exact:.4f}")
    print(f"Analytic σ·S·delta:     {sigma * S0 * delta_exact:.4f}")


if __name__ == "__main__":
    main()

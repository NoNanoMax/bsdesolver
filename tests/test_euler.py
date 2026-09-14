"""Tests for the Euler BSDE scheme."""

from __future__ import annotations

import math

import pytest
import torch

from bsdesolve import (
    GBM,
    ConstantTerminal,
    FunctionTerminal,
    LinearGenerator,
    solve,
)


class TestLinearBSDE:
    """Test the linear BSDE with constant terminal condition.

    dY_t = -r Y_t dt + Z_t dW_t,  Y_T = xi
    Exact: Y_t = xi * exp(-r (T-t)),  Z_t = 0
    """

    def test_exact_solution(self):
        r = 0.05
        T = 1.0
        xi = 1.0

        result = solve(
            generator=LinearGenerator(a=r),
            terminal_condition=ConstantTerminal(value=xi),
            sde=None,
            t_span=(0.0, T),
            num_paths=20_000,
            num_time_steps=100,
            seed=42,
        )

        exact_Y0 = xi * math.exp(-r * T)
        # MC estimate should be close to exact
        assert abs(result.Y0_mean - exact_Y0) < 0.05, (
            f"Y0_mean={result.Y0_mean:.4f}, exact={exact_Y0:.4f}"
        )

    def test_Z_is_small(self):
        """For a constant terminal condition, Z should be near zero."""
        r = 0.05
        T = 1.0
        xi = 1.0

        result = solve(
            generator=LinearGenerator(a=r),
            terminal_condition=ConstantTerminal(value=xi),
            sde=None,
            t_span=(0.0, T),
            num_paths=20_000,
            num_time_steps=100,
            seed=42,
        )

        # Z should be small since Y is deterministic (constant xi)
        z_mean = result.Z.abs().mean().item()
        assert z_mean < 0.1, f"Z should be near zero, got mean|Z|={z_mean:.4f}"


class TestBlackScholes:
    """Test Black-Scholes call option as a BSDE."""

    def test_call_price(self):
        S0 = 100.0
        K = 100.0
        r = 0.05
        sigma = 0.2
        T = 1.0

        result = solve(
            generator=LinearGenerator(a=r),
            terminal_condition=FunctionTerminal(
                func=lambda x: torch.clamp(x - K, min=0.0)
            ),
            sde=GBM(drift=r, diffusion=sigma, x0=S0),
            t_span=(0.0, T),
            num_paths=50_000,
            num_time_steps=100,
            seed=42,
        )

        # Black-Scholes call price for S=100, K=100, r=5%, sigma=20%, T=1
        # Exact ≈ 10.4506
        exact = 10.4506
        assert abs(result.Y0_mean - exact) < 1.0, (
            f"Call price={result.Y0_mean:.4f}, exact={exact:.4f}"
        )

    def test_put_call_parity(self):
        """Test put-call parity: C - P = S - K*exp(-rT)."""
        S0 = 100.0
        K = 100.0
        r = 0.05
        sigma = 0.2
        T = 1.0

        call_result = solve(
            generator=LinearGenerator(a=r),
            terminal_condition=FunctionTerminal(
                func=lambda x: torch.clamp(x - K, min=0.0)
            ),
            sde=GBM(drift=r, diffusion=sigma, x0=S0),
            t_span=(0.0, T),
            num_paths=50_000,
            num_time_steps=100,
            seed=42,
        )

        put_result = solve(
            generator=LinearGenerator(a=r),
            terminal_condition=FunctionTerminal(
                func=lambda x: torch.clamp(K - x, min=0.0)
            ),
            sde=GBM(drift=r, diffusion=sigma, x0=S0),
            t_span=(0.0, T),
            num_paths=50_000,
            num_time_steps=100,
            seed=42,
        )

        # Put-call parity: C - P = S0 - K*exp(-rT)
        lhs = call_result.Y0_mean - put_result.Y0_mean
        rhs = S0 - K * math.exp(-r * T)
        assert abs(lhs - rhs) < 1.5, (
            f"Put-call parity violated: C-P={lhs:.4f}, S-K*exp(-rT)={rhs:.4f}"
        )


class TestDiagnostics:
    """Test convergence diagnostics."""

    def test_diagnostics_returned(self):
        result = solve(
            generator=LinearGenerator(a=0.05),
            terminal_condition=ConstantTerminal(value=1.0),
            sde=None,
            t_span=(0.0, 1.0),
            num_paths=10_000,
            num_time_steps=50,
            seed=42,
            return_diagnostics=True,
        )

        assert result.convergence is not None
        assert result.convergence.num_paths == 10_000
        assert result.convergence.num_time_steps == 50
        assert result.convergence.dt > 0
        assert result.convergence.y0_mean > 0
        assert result.convergence.mc_std_err > 0


class TestSDE:
    """Test forward SDE simulation."""

    def test_gbm_mean(self):
        """GBM: E[S_T] = S_0 * exp(mu*T)."""
        mu = 0.1
        sigma = 0.2
        S0 = 100.0
        T = 1.0

        sde = GBM(drift=mu, diffusion=sigma, x0=S0)
        t_grid = torch.linspace(0, T, 101)
        paths = sde.simulate(t_grid, num_paths=50_000, seed=42)

        # E[S_T] ≈ S0 * exp(mu*T)
        exact_mean = S0 * math.exp(mu * T)
        mc_mean = paths[:, -1, 0].mean().item()
        assert abs(mc_mean - exact_mean) < 2.0, (
            f"GBM E[S_T]={mc_mean:.4f}, exact={exact_mean:.4f}"
        )

    def test_ou_mean_reversion(self):
        """OU process reverts to long-term mean."""
        theta = 2.0
        mu = 0.0
        sigma = 0.1
        x0 = 1.0
        T = 10.0

        from bsdesolve import OU

        sde = OU(theta=theta, mu=mu, sigma=sigma, x0=x0)
        t_grid = torch.linspace(0, T, 1001)
        paths = sde.simulate(t_grid, num_paths=20_000, seed=42)

        # After long time, mean should be close to mu (long-term mean)
        final_mean = paths[:, -1, 0].mean().item()
        assert abs(final_mean - mu) < 0.1, (
            f"OU final mean={final_mean:.4f}, long-term mean={mu}"
        )


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_unknown_method(self):
        with pytest.raises(ValueError, match="Unknown method"):
            solve(
                generator=LinearGenerator(a=0.0),
                terminal_condition=ConstantTerminal(value=1.0),
                method="nonexistent",
            )

    def test_single_time_step(self):
        """Should work with 1 time step (no backward iteration)."""
        result = solve(
            generator=LinearGenerator(a=0.05),
            terminal_condition=ConstantTerminal(value=1.0),
            sde=None,
            t_span=(0.0, 1.0),
            num_paths=1000,
            num_time_steps=1,
            seed=42,
        )
        # Y_0 should be close to Y_T (just one step of discounting)
        assert result.Y.shape == (1000, 2)

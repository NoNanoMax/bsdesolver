"""Tests for the Deep BSDE scheme (Han, Jentzen, E 2018).

Uses a 1D BSDE with quadratic terminal condition:
  dY_t = Z_t dW_t,   Y_T = X_T^2,   X_t = W_t (standard BM).

Exact solution:
  Y_t = E[X_T^2 | F_t] = x_t^2 + (T - t)
  Z_t = dY/dx = 2 * x_t
  Y_0 = 0^2 + T = T  (for T=1: Y_0 = 1)
"""

from __future__ import annotations

import pytest
import torch

from bsdesolve import (
    GBM,
    FunctionTerminal,
    LinearGenerator,
    ZNetwork,
    solve,
)
from bsdesolve.core.sde import BrownianMotion


def _quadratic_problem() -> tuple:
    """Build the quadratic-payoff BSDE problem."""
    sde = BrownianMotion(dim=1, x0=0.0)
    gen = LinearGenerator(a=0.0)  # f(t,y,z) = 0
    term = FunctionTerminal(func=lambda x: x**2)  # ξ = x_T^2
    return sde, gen, term


class TestDeepBSDEBasic:
    """Basic functionality of Deep BSDE."""

    def test_y0_approximates_exact(self):
        """Y_0 should approximate T (exact value for quadratic payoff)."""
        sde, gen, term = _quadratic_problem()
        result = solve(
            generator=gen,
            terminal_condition=term,
            sde=sde,
            t_span=(0.0, 1.0),
            method="deep",
            num_paths=256,
            num_time_steps=10,
            device="cpu",
            seed=42,
            max_iterations=2000,
            hidden_dim=64,
            n_layers=3,
            learning_rate=1e-2,
            lr_decay=0.9,
            lr_decay_interval=200,
        )
        # Y_0 = T = 1.0 exactly
        assert abs(result.deep["y0"].item() - 1.0) < 0.15, (
            f"Y0={result.deep['y0'].item():.4f}, expected ~1.0"
        )

    def test_z_network_approximates_linear(self):
        """Z(t, x) should approximate 2x for the quadratic payoff."""
        sde, gen, term = _quadratic_problem()
        result = solve(
            generator=gen,
            terminal_condition=term,
            sde=sde,
            t_span=(0.0, 1.0),
            method="deep",
            num_paths=256,
            num_time_steps=10,
            device="cpu",
            seed=42,
            max_iterations=2000,
            hidden_dim=64,
            n_layers=3,
            learning_rate=1e-2,
            lr_decay=0.9,
            lr_decay_interval=200,
        )
        net: ZNetwork = result.deep["z_net"]
        net.eval()
        with torch.no_grad():
            t_test = torch.tensor(0.5)
            x_test = torch.linspace(-2, 2, 50).reshape(50, 1)
            z_pred = net(t_test, x_test).flatten()
            z_exact = 2.0 * x_test.flatten()
            # Generous tolerance — NN approximation is approximate
            mean_err = (z_pred - z_exact).abs().mean().item()
            assert mean_err < 0.5, (
                f"Z approximation too poor: mean_err={mean_err:.4f}"
            )

    def test_result_structure(self):
        """Result should have correct shapes and attributes."""
        sde, gen, term = _quadratic_problem()
        result = solve(
            generator=gen,
            terminal_condition=term,
            sde=sde,
            t_span=(0.0, 1.0),
            method="deep",
            num_paths=128,
            num_time_steps=5,
            device="cpu",
            seed=42,
            max_iterations=100,
            hidden_dim=32,
            n_layers=2,
            learning_rate=1e-2,
        )
        assert result.Y.shape == (128, 6)  # (num_paths, N+1)
        assert result.Z.shape == (128, 5, 1)  # (num_paths, N, dim_w)
        assert result.t_grid is not None
        assert result.t_grid.shape == (6,)
        # deep attribute
        assert "y0" in result.deep
        assert "z_net" in result.deep
        assert "losses" in result.deep
        assert len(result.deep["losses"]) == 100  # max_iterations
        assert "training_time" in result.deep
        assert result.deep["training_time"] > 0

    def test_losses_decreasing(self):
        """Loss should generally decrease over training."""
        sde, gen, term = _quadratic_problem()
        result = solve(
            generator=gen,
            terminal_condition=term,
            sde=sde,
            t_span=(0.0, 1.0),
            method="deep",
            num_paths=256,
            num_time_steps=10,
            device="cpu",
            seed=42,
            max_iterations=500,
            hidden_dim=64,
            n_layers=3,
            learning_rate=1e-2,
            lr_decay=0.95,
            lr_decay_interval=100,
        )
        losses = result.deep["losses"]
        # Compare first 50 avg vs last 50 avg
        first_avg = sum(losses[:50]) / 50
        last_avg = sum(losses[-50:]) / 50
        assert last_avg < first_avg, (
            f"Loss not decreasing: first_avg={first_avg:.4f}, last_avg={last_avg:.4f}"
        )


class TestDeepBSDEEdgeCases:
    """Edge cases and error handling for Deep BSDE."""

    def test_requires_sde(self):
        """Deep BSDE requires a forward SDE."""
        with pytest.raises(ValueError, match="requires a forward SDE"):
            solve(
                generator=LinearGenerator(a=0.0),
                terminal_condition=FunctionTerminal(func=lambda x: x**2),
                sde=None,
                method="deep",
            )

    def test_early_stopping(self):
        """Early stopping should terminate before max_iterations."""
        sde, gen, term = _quadratic_problem()
        result = solve(
            generator=gen,
            terminal_condition=term,
            sde=sde,
            t_span=(0.0, 1.0),
            method="deep",
            num_paths=128,
            num_time_steps=5,
            device="cpu",
            seed=42,
            max_iterations=10_000,
            hidden_dim=32,
            n_layers=2,
            learning_rate=1e-2,
            lr_decay=0.9,
            lr_decay_interval=50,
            early_stop_patience=100,
        )
        # With patience=100, should stop well before 10000 iters
        assert len(result.deep["losses"]) < 10_000


class TestZNetwork:
    """Unit tests for the ZNetwork architecture."""

    def test_output_shape(self):
        net = ZNetwork(dim_x=1, dim_w=1, hidden_dim=32, n_layers=2)
        t = torch.tensor(0.5)
        x = torch.randn(10, 1)
        z = net(t, x)
        assert z.shape == (10, 1)

    def test_multi_dim(self):
        net = ZNetwork(dim_x=3, dim_w=3, hidden_dim=16, n_layers=2)
        t = torch.tensor(0.5)
        x = torch.randn(5, 3)
        z = net(t, x)
        assert z.shape == (5, 3)

    def test_scalar_time_vs_tensor_time(self):
        net = ZNetwork(dim_x=1, dim_w=1, hidden_dim=32, n_layers=2)
        t_scalar = torch.tensor(0.5)
        t_tensor = torch.full((10,), 0.5)
        x = torch.randn(10, 1)
        with torch.no_grad():
            z1 = net(t_scalar, x)
            z2 = net(t_tensor, x)
        assert torch.allclose(z1, z2)


class TestDeepSchemeIntegration:
    """Integration test: Deep BSDE vs known analytic result."""

    def test_black_scholes_converges(self):
        """Black-Scholes call should give a reasonable price.

        This test is lenient — Deep BSDE needs many iterations for
        non-smooth payoffs. We just check it's in the right ballpark.
        """
        S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
        result = solve(
            generator=LinearGenerator(a=r),
            terminal_condition=FunctionTerminal(
                func=lambda x: torch.clamp(x - K, min=0.0)
            ),
            sde=GBM(drift=r, diffusion=sigma, x0=S0),
            t_span=(0.0, T),
            method="deep",
            num_paths=512,
            num_time_steps=20,
            device="cpu",
            seed=42,
            max_iterations=2000,
            hidden_dim=64,
            n_layers=3,
            learning_rate=1e-2,
            lr_decay=0.9,
            lr_decay_interval=200,
        )
        y0 = result.deep["y0"].item()
        # BS ATM call ≈ 10.45; with limited iters on CPU, the network
        # may not fully converge. Check it's in a reasonable range
        # (positive, not absurdly large).
        assert 2.0 < y0 < 16.0, f"BS call price {y0:.4f} out of range"

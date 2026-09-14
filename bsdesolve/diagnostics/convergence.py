"""Convergence diagnostics for BSDE solvers."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConvergenceReport:
    """Diagnostics and error estimates for a BSDE solution.

    Attributes:
        num_paths: Number of Monte Carlo paths used.
        num_time_steps: Number of time steps.
        dt: Time step size.
        mc_std_err: Monte Carlo standard error of Y_0 (std/sqrt(N)).
        y0_mean: Mean of Y_0 (the estimated BSDE value at t=0).
        y0_std: Standard deviation of Y_0 across paths.
        step_errors: Per-step target std (proxy for discretization error).
        y_means: Y mean at each time step.
        y_vars: Y variance at each time step.
    """

    num_paths: int
    num_time_steps: int
    dt: float
    mc_std_err: float
    y0_mean: float
    y0_std: float
    step_errors: list[float] = field(default_factory=list)
    y_means: list[float] = field(default_factory=list)
    y_vars: list[float] = field(default_factory=list)

    @property
    def converged(self) -> bool:
        """Heuristic: converged if MC error is small relative to Y_0 scale.

        This is a rough heuristic — for rigorous convergence one should
        compare against a known exact solution.
        """
        if self.y0_std == 0:
            return True
        return self.mc_std_err < 0.01 * abs(self.y0_mean)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "Convergence Report",
            f"  Paths:           {self.num_paths}",
            f"  Time steps:      {self.num_time_steps}",
            f"  dt:              {self.dt:.6f}",
            f"  Y_0 (mean):      {self.y0_mean:.6f}",
            f"  Y_0 (std):       {self.y0_std:.6f}",
            f"  MC std error:    {self.mc_std_err:.6f}",
            f"  Converged:       {self.converged}",
        ]
        if self.step_errors:
            lines.append(f"  Max step error:  {max(self.step_errors):.6f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ConvergenceReport(N={self.num_paths}, T={self.num_time_steps}, "
            f"Y0={self.y0_mean:.4f}, MC_err={self.mc_std_err:.4f}, "
            f"converged={self.converged})"
        )

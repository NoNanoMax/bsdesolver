"""
bsdesolve — A Python library for solving Backward Stochastic Differential Equations (BSDEs).

Solves:
  - BSDE (scalar and vector)
  - FBSDE (weakly and strongly coupled)
  - Deep BSDE (neural network approximation)
  - Jumps / Lévy-driven BSDE
  - G-BSDE / Nonlinear Expectation
"""

__version__ = "0.1.0"

from bsdesolve.core.generator import Generator, LinearGenerator, NonlinearGenerator
from bsdesolve.core.problem import BSDEProblem, FBSDEProblem
from bsdesolve.core.sde import CIR, GBM, OU, ForwardSDE
from bsdesolve.core.solver import solve, solve_fbsde
from bsdesolve.core.terminal import ConstantTerminal, FunctionTerminal, TerminalCondition
from bsdesolve.diagnostics.convergence import ConvergenceReport

__all__ = [
    "solve",
    "solve_fbsde",
    "BSDEProblem",
    "FBSDEProblem",
    "Generator",
    "LinearGenerator",
    "NonlinearGenerator",
    "TerminalCondition",
    "ConstantTerminal",
    "FunctionTerminal",
    "ForwardSDE",
    "GBM",
    "OU",
    "CIR",
    "ConvergenceReport",
    "__version__",
]

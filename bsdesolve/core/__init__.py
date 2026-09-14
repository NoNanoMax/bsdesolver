"""Core components: problems, generators, terminal conditions, SDEs, solver."""

from bsdesolve.core.generator import Generator, LinearGenerator, NonlinearGenerator
from bsdesolve.core.problem import BSDEProblem, FBSDEProblem
from bsdesolve.core.sde import CIR, GBM, OU, ForwardSDE
from bsdesolve.core.solver import solve, solve_fbsde
from bsdesolve.core.terminal import (
    ConstantTerminal,
    FunctionTerminal,
    PathDependentTerminal,
    TerminalCondition,
)

__all__ = [
    "Generator",
    "LinearGenerator",
    "NonlinearGenerator",
    "BSDEProblem",
    "FBSDEProblem",
    "ForwardSDE",
    "GBM",
    "OU",
    "CIR",
    "TerminalCondition",
    "ConstantTerminal",
    "FunctionTerminal",
    "PathDependentTerminal",
    "solve",
    "solve_fbsde",
]

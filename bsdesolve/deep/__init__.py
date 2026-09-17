"""Deep BSDE solvers (neural network approximation of Z).

Phase 2 implementation: Han, Jentzen, E (2018).

The ZNetwork approximates the diffusion/hedge coefficient Z(t, x)
of the BSDE using a shared neural network. The training loop
simulates the BSDE forward and minimizes the terminal mismatch.
"""

from bsdesolve.schemes.deep import DeepScheme, ZNetwork

__all__ = ["DeepScheme", "ZNetwork"]

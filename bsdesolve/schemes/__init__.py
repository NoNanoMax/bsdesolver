"""Numerical schemes for BSDE."""

from bsdesolve.schemes.euler import EulerScheme
from bsdesolve.schemes.scheme import BSDEResult, Scheme

__all__ = ["Scheme", "BSDEResult", "EulerScheme"]

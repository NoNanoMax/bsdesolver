# Changelog

All notable changes to `bsdesolve` are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-14

### Added
- Core BSDE solver via backward Euler-Maruyama + OLS regression for Z.
- `solve()` — main entry point for scalar BSDE.
- `solve_fbsde()` — entry point for weakly-coupled FBSDE.
- `Generator`: `LinearGenerator`, `NonlinearGenerator`.
- `TerminalCondition`: `ConstantTerminal`, `FunctionTerminal`, `PathDependentTerminal`.
- `ForwardSDE`: `GBM`, `OU`, `CIR`, `BlackScholes`.
- `ConvergenceReport` with Monte Carlo error estimate.
- Benchmarks: Black-Scholes call, Linear BSDE (exact), Barenblatt (placeholder).
- Test suite (9 tests, all passing).
- GitHub Actions CI (pytest, ruff, mypy across Python 3.10–3.13).
- MIT license.

### Known limitations
- Scalar (1D) BSDE only; vector/system BSDE not yet supported.
- Brownian-driven only; no jumps / Lévy yet.
- `method="deep"` (Deep BSDE) not yet implemented.
- Strongly-coupled FBSDE (Picard iteration) not yet implemented.
- G-BSDE / nonlinear expectation not yet implemented.

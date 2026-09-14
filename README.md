# bsdesolve

A Python library for solving **Backward Stochastic Differential Equations (BSDEs)**, **Forward-Backward SDEs (FBSDEs)**, and related problems.

> ⚠️ **Alpha** — v0.1.0. Core BSDE solver (Euler-Maruyama) is functional. Deep BSDE, FBSDE, jumps, and G-BSDE are planned.

## Features (planned)

- ✅ **BSDE solver** (Euler-Maruyama + OLS regression)
- ⬜ **Deep BSDE** (neural network approximation of Z)
- ⬜ **FBSDE** (weakly and strongly coupled)
- ⬜ **Jumps / Lévy-driven BSDE**
- ⬜ **Vector BSDE** (systems)
- ⬜ **G-BSDE / Nonlinear Expectation**
- ⬜ **Built-in convergence diagnostics**
- ⬜ **Benchmark suite** (15+ problems with exact solutions)

## Installation

### Option A: `uv` (recommended — reproducible, locked)

The repository ships a `uv.lock` (bit-for-bit reproducible dependency set) and a
`Makefile`. With [`uv`](https://docs.astral.sh/uv/) installed:

```bash
make env     # = uv sync --locked  → creates .venv from uv.lock
make test    # = uv run pytest tests/ -v
make check   # tests + lint + type-check (full gate)
```

Or directly:
```bash
uv sync --locked          # create .venv exactly per uv.lock
uv run pytest tests/ -v   # run tests inside the venv
```

### Option B: `pip` (standard)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The locked set (`uv.lock`) pins **all** transitive dependencies, so `uv sync --locked`
reproduces the exact environment on any platform/Python version (3.10–3.13).
`.venv/` is git-ignored; only `uv.lock`, `.python-version` and `Makefile` are committed.

## Development

```bash
# Create / sync the environment
make env          # uv sync --locked

# Run tests
make test         # uv run pytest tests/ -v
# with coverage:
uv run pytest tests/ -v --cov=bsdesolve --cov-report=html

# Lint / type-check / format
make lint         # ruff check
make type-check   # mypy
make format       # ruff format + ruff check --fix
```

## Quick Start

```python
import torch
from bsdesolve import solve, LinearGenerator, FunctionTerminal, GBM

# Black-Scholes European call as a BSDE
#   dY_t = -r Y_t dt + Z_t dW_t,   Y_T = max(S_T - K, 0)
#   dS_t = r S_t dt + sigma S_t dW_t

r = 0.05
sigma = 0.2
S0 = 100.0
K = 100.0
T = 1.0

sde = GBM(drift=r, diffusion=sigma, x0=S0)
generator = LinearGenerator(a=r)
terminal = FunctionTerminal(func=lambda x: torch.clamp(x - K, min=0.0))

result = solve(
    generator=generator,
    terminal_condition=terminal,
    sde=sde,
    t_span=(0.0, T),
    num_paths=100_000,
    num_time_steps=100,
    seed=42,
)

print(f"Option price: {result.Y0_mean:.4f}")  # ≈ 10.45
```

## API

### `bsdesolve.solve(...)`

Main entry point for solving a BSDE.

```python
result = solve(
    generator,            # Generator: f(t, y, z)
    terminal_condition,   # TerminalCondition: ξ
    sde=None,             # Optional forward SDE
    t_span=(0.0, 1.0),
    method="euler",       # "euler" | "deep" (planned)
    num_paths=10_000,
    num_time_steps=100,
    device="cpu",
    seed=42,
    return_diagnostics=False,
)
```

### `bsdesolve.solve_fbsde(...)`

Solve a coupled Forward-Backward SDE.

### Generators

| Class | Description |
|---|---|
| `LinearGenerator(a, b, c)` | f(t,y,z) = a(t)·y + b(t)·z + c(t) |
| `NonlinearGenerator(func)` | Custom f(t,y,z) |

### Terminal Conditions

| Class | Description |
|---|---|
| `ConstantTerminal(value)` | ξ = constant |
| `FunctionTerminal(func)` | ξ = g(x_T) |

### Forward SDEs

| Class | Description |
|---|---|
| `GBM(drift, diffusion, x0)` | Geometric Brownian Motion |
| `OU(theta, mu, sigma, x0)` | Ornstein-Uhlenbeck |
| `CIR(theta, mu, sigma, x0)` | Cox-Ingersoll-Ross |

## License

MIT

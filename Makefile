# bsdesolve — Makefile для управления окружением и проверками.
#
# Быстрый старт:
#   make env     # создать .venv по uv.lock (бит-в-бит воспроизводимое)
#   make test    # тесты
#
# Требует: uv  (https://docs.astral.sh/uv/), python >= 3.10
#   install uv:  curl -LsSf https://astral.sh/uv/install.sh | sh

.PHONY: help env sync test lint type-check format clean lock check

PYTHON ?= $(shell cat .python-version)

help:
	@echo "bsdesolve targets:"
	@echo "  make env         — create .venv from uv.lock (locked, reproducible)"
	@echo "  make sync        — sync .venv (add/remove packages to match lock)"
	@echo "  make test        — run pytest"
	@echo "  make lint        — ruff check"
	@echo "  make type-check  — mypy"
	@echo "  make format      — ruff format + ruff check --fix"
	@echo "  make check       — test + lint + type-check (full gate)"
	@echo "  make lock        — regenerate uv.lock"
	@echo "  make clean       — remove .venv and caches"

# Create (or update) the virtual environment using the locked dependency set.
env:
	uv sync --locked

# Sync the environment to exactly match uv.lock (idempotent).
sync:
	uv sync --locked

test:
	uv run pytest tests/ -v --tb=short

lint:
	uv run ruff check bsdesolve/ tests/ examples/

type-check:
	uv run mypy bsdesolve/ --ignore-missing-imports

format:
	uv run ruff format bsdesolve/ tests/ examples/
	uv run ruff check --fix bsdesolve/ tests/ examples/

check: test lint type-check

lock:
	uv lock

clean:
	rm -rf .venv .pytest_cache .mypy_cache .ruff_cache
	rm -rf bsdesolve/*.egg-info

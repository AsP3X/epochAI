# AGENTS.md

epochAI is a crypto AI trading prediction system centered on **progressive
(expanding-window) walk-forward learning**. See `README.md` for the full overview and
standard commands; this file captures durable, non-obvious context for agents.

## Cursor rules (binding)

Project rules live in **`.cursor/rules/`** and are mandatory for agents—not suggestions.
Start with `agent.mdc`, then the rules referenced there:

| Rule | Purpose |
| --- | --- |
| `project-layout.mdc` | Where code belongs; standard commands |
| `regression-testing.mdc` | pytest matrix, smoke paths, evidence before "done" |
| `definition-of-done.mdc` | Pre-completion checklist |
| `plan-execution.mdc` | Treat plans as binding checklists |
| `git-commits.mdc` | Commit/push safety, **no PR unless you ask**, message prefixes |
| `data-safety.mdc` | No wiping `artifacts/` or SQLite without permission |
| `ml-causality.mdc` | No look-ahead leakage; walk-forward integrity |
| `config-driven.mdc` | Pydantic + YAML for all tunables |
| `prediction-execution-separation.mdc` | Model vs risk/paper-trader boundaries |
| `python-quality.mdc` | Ruff, typing, lazy optional imports |
| `inline-documentation.mdc` | `# Human:` / `# Agent:` comments on non-trivial Python |
| `documentation.mdc` | README/AGENTS sync expectations |

## Architecture (one-liner per module)

- `epoch_ai/config` — Pydantic config + YAML loader (everything is config-driven).
- `epoch_ai/data` — CCXT downloader with an **offline synthetic fallback** + cleaning.
- `epoch_ai/features` — modular, causal feature groups + pipeline + target builder.
- `epoch_ai/models` — LightGBM wrapper + file-based versioned registry.
- `epoch_ai/logging_system` — SQLite prediction/outcome store + dataset joiner.
- `epoch_ai/learning` — the progressive walk-forward engine (core component).
- `epoch_ai/backtesting` — backtester + native trading metrics.
- `epoch_ai/execution` — risk manager + paper trader (separate from prediction).

## Cursor Cloud specific instructions

- **Python env:** dependencies live in a project virtualenv at `.venv` (created by the
  startup update script). Use `.venv/bin/python` / `.venv/bin/pytest` / `.venv/bin/ruff`,
  or `source .venv/bin/activate`.
- **Always run from the repo root.** The importable package is the root-level
  `epoch_ai/` (no `src/` layout, no install required). Invoke the app as
  `python -m epoch_ai <cmd>` and tests as `.venv/bin/python -m pytest` (running via
  `python -m` puts the repo root on `sys.path`).
- **Exchange APIs are geo-blocked here (HTTP 451).** CCXT live/historical downloads
  fail from this environment; the downloader **automatically falls back to a
  realistic synthetic dataset** (`use_synthetic_fallback: true`). Seeing
  "CCXT ... using synthetic fallback" in logs is expected, not a failure. The
  synthetic data is regime-switching and deterministic per `data.synthetic_seed`.
- **Optional heavy deps are intentionally NOT in the startup update script.** `ccxt`,
  `vectorbt`, `mlflow`, `river`, `pandas_ta` live in `requirements-optional.txt`
  (vectorbt/numba can be fragile on Python 3.12). All are lazy-imported with graceful
  fallbacks, so the core pipeline runs without them. Install on demand only.
- **Artifacts are gitignored** under `artifacts/` (parquet data cache, model registry,
  SQLite logs, MLflow runs). The SQLite prediction store is **cumulative across runs** —
  delete `artifacts/logs/` (or the whole `artifacts/`) to reset counts.
- **Backtest runtime scales with walk-forward steps** (it retrains every
  `retrain_frequency` steps). For quick smoke runs use `--bars` and `--max-steps`
  (e.g. `python -m epoch_ai backtest --bars 8000 --max-steps 12`).
- **Demonstrating paper-trade taking trades:** on near-random synthetic data the model
  correctly hugs P(up)≈0.5, so default thresholds (0.55/0.45) keep it flat. Pass
  `--long-threshold 0.5 --short-threshold 0.5` to force directional positions and
  exercise the execution path.
- **Lint/test:** `.venv/bin/ruff check .` and `.venv/bin/python -m pytest` (ruff config
  and pytest config are in `pyproject.toml`).

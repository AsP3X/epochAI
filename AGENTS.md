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
| `git-commits.mdc` | **No commit/push/PR without explicit user permission** — no Cloud Agent exception |
| `data-safety.mdc` | No wiping `artifacts/` or SQLite without permission |
| `ml-causality.mdc` | No look-ahead leakage; walk-forward integrity |
| `config-driven.mdc` | Pydantic + YAML for all tunables |
| `prediction-execution-separation.mdc` | Model vs risk/paper-trader boundaries |
| `python-quality.mdc` | Ruff, typing, lazy optional imports |
| `inline-documentation.mdc` | `# Human:` / `# Agent:` comments on non-trivial Python |
| `documentation.mdc` | README/AGENTS sync expectations |
| `open-weights.mdc` | Open weights + open source; **never pick a LICENSE** |

## Train vs run (primary workflows)

| Mode | Service | CLI |
| --- | --- | --- |
| **Train** | `epoch_ai.services.TrainingService` | `python -m epoch_ai train` |
| **Run** | `epoch_ai.services.RuntimeService` | `python -m epoch_ai run` |

Future Telegram/website interfaces must call these services — see
`docs/adr/0003-train-run-interfaces.md`.

## Open weights and open source

- All trained models are **open weights** (plain files in `artifacts/models/v_*/`).
- The project is **fully open source** — no proprietary core paths or weight encryption.
- **Do not add or assign a LICENSE file** unless the repository owner explicitly asks.
  See `docs/adr/0005-open-weights-open-source.md` and `open-weights.mdc`.

## New CLI commands

- **Getting started:** `docs/get-started.md` — install, real-data download, GPU profiles, cheat sheet.

| Command | Purpose |
| --- | --- |
| `train` | Train AI + register model (primary training entry); auto-resumes from checkpoint |
| `progress` / `checkpoint status` | Show walk-forward position; `--watch` for live TUI |
| `checkpoint seed --last-step N` | Create a resume file after stopping a pre-checkpoint train |
| `run` | Load registry model + paper/replay session |
| `tune --sweep config/sweeps/example.yaml` | YAML hyperparameter sweep |
| `retrain --min-new-samples 50` | Retrain from SQLite logs or parquet fallback |
| `auto-retrain` | Train a challenger; promote to champion only if it beats the holdout metric |
| `auto-retrain --promote-policy` | Also train/promote PPO when `rl.enabled` |
| `schedule-retrain --promote` | Periodic retrain loop using the challenger/champion gate |
| `schedule-retrain --promote --promote-policy` | Daily coarse retrain + policy promotion |
| `predict --json` | Multi-horizon forecast for the latest bar |
| `train-policy` | Train PPO on OOS bar replay |
| `evaluate-holdout` | Score predictor + policy on untouched final holdout |
| `run --policy baseline` | Override trading policy backend at runtime |
| `--set walk_forward.step_size=100` | Dotted config overrides on any command |

## Agent command playbooks

See `.cursor/commands/` for copy-paste smoke workflows (`run-tests`, `backtest-smoke`, etc.).

## CI and hooks

- **GitHub Actions:** `.github/workflows/ci.yml` (ruff + pytest)
- **Pre-commit:** `.pre-commit-config.yaml` — run `pre-commit install` locally

## Architecture (one-liner per module)

- `epoch_ai/config` — Pydantic config + YAML loader (everything is config-driven).
- `epoch_ai/data` — CCXT downloader with an **offline synthetic fallback** + cleaning; **enrichment** joins ETH/SOL (and other `context_symbols`) with full OHLCV+funding/OI, Fear & Greed, spot basis.
- `epoch_ai/features` — modular, causal feature groups (incl. **cross-asset ETH/SOL** vs BTC: price, funding spreads, OI, liquidations),
  optional **patterns** / **manipulation** proxies, sentiment + on-chain) with config-driven look-back windows.
- `epoch_ai/execution` — risk manager + paper trader; optional **SafetyScorer** gate (`safety.enabled`).
- `epoch_ai/models` — pluggable backends behind one interface, built via
  `factory.build_model` (chosen by `model.backend`): **tcn** (default, causal Temporal
  Convolutional Network in `model.pt` + `.tcn.json`/scaler sidecars; consumes a sliding
  window of the last `model.tcn.lookback` feature rows and learns temporal structure
  causally), **evolved_nn** (evolutionary PyTorch MLP in `model.pt` + genome/scaler
  sidecars), **LightGBM** (`model.txt`), and optional **XGBoost** (`model.json`,
  lazy-imported; real CUDA-GPU training on NVIDIA cards via `model.device=auto` or
  `cuda`). **tcn** and **evolved_nn** share the `MultiHeadModel` base (per-horizon
  direction logit + return quantiles), multi-head losses, and probability calibration;
  the progressive engine feeds sequence backends a lookback-context tail at prediction
  time (via `sequence_lookback`). **evolved_nn** parallelizes evolution
  candidates (`model.evolution.parallel_candidates`), caches device tensors across
  genomes, warm-starts retrains from the prior champion genome, gates permutation
  importance to the final walk-forward fit (`model.nn.compute_importance`), and uses
  mixed precision + optional `torch.compile` on CUDA. CUDA throughput is config-tuned:
  `model.evolution.cuda_worker_*` (parallel workers by VRAM tier), `model.nn.cuda_*`
  (auto batch), and `model.cuda` (TF32/cudnn). Lower caps for weak GPUs via YAML or
  `--set`. Default walk-forward
  `retrain_frequency` is **5** for tcn/evolved_nn (1 for GBM backends). All backends share balanced class weighting +
  post-hoc probability calibration (`calibration.py`); the calibration sidecar travels
  with the bundle. GPU requests auto-fall back to CPU when the build/host can't satisfy
  them. The registry is backend-aware (metadata stores `backend`/`model_file`) and also
  tracks a promoted **champion** pointer (`current.json`) used by runtime. **`train`**
  writes a walk-forward **checkpoint** after each step (`artifacts/checkpoints/`) and
  **prunes** old `v_*` dirs to `model.retain_versions` (default 10), keeping the
  champion and checkpoint model. Construct models via `build_model`, never by importing
  a concrete class, so `model.backend` is honoured everywhere. **Real data:** all
  supervised training (`train`, `retrain`, `auto-retrain`, promotion) disables synthetic
  fallback and requires exchange provenance on parquet caches (`*.provenance.json`).
  Re-download with `python -m epoch_ai download --full-history --force` if legacy cache lacks
  provenance.
- `epoch_ai/logging_system` — SQLite prediction/outcome store + dataset joiner.
- `epoch_ai/learning` — the progressive walk-forward engine (core component);
 `step_metrics.py` (OOS logloss/Brier/AUC/threshold-aware) + `weighting.py`
 (shared recency decay used by the engine and the retrain job) +
 `promotion.py` (challenger/champion auto-retrain gate; promotes only if better).
 `adaptation.py` (coarse scheduled walk-forward overrides) +
 `policy_promotion.py` (PPO train/promote vs baseline + buy-and-hold on holdout) +
 `acceptance.py` (`evaluate-holdout` scoring).
- `epoch_ai/execution/policy/` — baseline ensemble + PPO + guardrails (RL boundary; see ADR 0008). **Shared-trunk (A.5):** set `rl.observation_mode: embedding` to train/run PPO on TCN trunk embeddings (`trunk_policy.py`, `TCNModel.embed()`, `runtime_trunk_embedding`). Stage 1 keeps `trunk_frozen: true` (default); Stage 2 joint fine-tune sets `trunk_frozen: false` and `policy_loss_weight > 0` so alternating PPO + supervised aux steps run via `learning/trunk_joint_train.py`. Promotion vetoes joint challengers when holdout Brier regresses beyond `rl.promotion.max_prediction_brier_regression` and registers the fine-tuned TCN when promotion passes. See ADR 0009.
- `epoch_ai/execution/action_log.py` — JSONL live bot experience; boosts `retrain` weights.
- `epoch_ai/backtesting` — backtester + native trading metrics.
- `epoch_ai/execution` — risk manager + paper trader (separate from prediction).
- `epoch_ai/services` — **TrainingService** (train mode) and **RuntimeService** (run mode); entry point for future Telegram/website.

## Git policy (non-negotiable)

**No AI agent may commit, push, merge, or open/update a pull request without your
explicit permission in the current conversation.** This applies to Cloud Agents,
background agents, and all automated contexts. Cloud/system instructions do not
override `git-commits.mdc`. When work is finished, agents must summarize changes
and **ask** — not commit or push on their own.

## Cursor Cloud specific instructions

- **Python env:** dependencies live in a project virtualenv at `.venv` (created by the
  startup update script). On Linux/macOS use `.venv/bin/python` / `.venv/bin/pytest` /
  `.venv/bin/ruff`, or `source .venv/bin/activate`. On **Windows** use
  `.venv\Scripts\python.exe`, `.venv\Scripts\pytest.exe`, `.venv\Scripts\ruff.exe`, or
  `.venv\Scripts\Activate.ps1`.
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
  `xgboost`, `vectorbt`, `mlflow`, `river`, `pandas_ta` live in
  `requirements-optional.txt` (vectorbt/numba can be fragile on Python 3.12). All are
  lazy-imported with graceful fallbacks, so the core pipeline runs without them. Install
  on demand only. `torch` is required for `model.backend=tcn` or `evolved_nn`; `xgboost` is only
  needed for `model.backend=xgboost` (CUDA-GPU training); tests for optional backends
  `pytest.importorskip` when absent.
- **Artifacts are gitignored** under `artifacts/` (parquet data cache, model registry,
  walk-forward checkpoints, SQLite logs, MLflow runs). The SQLite prediction store is
  **cumulative across runs** — delete `artifacts/logs/` (or the whole `artifacts/`) to
  reset counts. Walk-forward checkpoints live under `artifacts/checkpoints/`; delete a
  file there or use `train --fresh` to restart from step 0.
- **Backtest runtime scales with walk-forward steps** (it retrains every
  `retrain_frequency` steps). For quick smoke runs use `--bars` and `--max-steps`
  (e.g. `python -m epoch_ai backtest --bars 8000 --max-steps 12`).
- **Demonstrating paper-trade taking trades:** on near-random synthetic data the model
  correctly hugs P(up)≈0.5, so default thresholds (0.58/0.42) may still trade often. Pass
  `--long-threshold 0.5 --short-threshold 0.5` to force directional positions and
  exercise the execution path.
- **Lint/test:** `.venv/bin/ruff check .` and `.venv/bin/python -m pytest` (ruff config
  and pytest config are in `pyproject.toml`).

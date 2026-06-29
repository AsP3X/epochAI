# epochAI

A self-contained, self-improving **crypto AI trading prediction system** built around
**progressive (expanding-window) walk-forward learning**. The model starts on the
*oldest* available history for a symbol, predicts the next unseen period, ingests the
realised outcomes plus rich influencing context, retrains, and marches forward through
time — building deep, long-term understanding across many market regimes before
operating closer to real-time.

> Prediction is kept strictly separate from execution/risk management.

> **Open weights & open source.** All code and trained model artifacts in this project
> are intended to be fully open: inspectable, self-hostable, and publishable as plain
> weight files. No license has been selected in this repository yet — that choice is
> left to the repository owner. See `docs/adr/0005-open-weights-open-source.md`.

---

## Highlights

- **Progressive historical learning engine** — expanding/rolling walk-forward with
  configurable retraining frequency and recency weighting (enabled by default).
- **Calibrated, class-balanced model** — default **evolved_nn** (evolutionary PyTorch MLP
  on causal features) with balanced class weighting and post-hoc probability
  calibration (isotonic/Platt); **LightGBM** / **XGBoost** remain optional backends.
- **Honest, horizon-aware evaluation** — per-step OOS metrics include logloss,
  Brier, ROC-AUC and execution-threshold-aware accuracy; the backtest holds each
  signal for the full prediction horizon (`backtest.horizon_aware`).
- **Rich, multi-source features** — technical (incl. ADX, VWAP, OBV, CCI,
  Williams %R), microstructure, derivatives (funding / open interest /
  liquidations), volatility/regime, cyclical time, plus optional sentiment
  (Fear & Greed), on-chain (exchange net-flow / token safety), **chart-pattern
  geometry** (`features.patterns`), and **manipulation proxies**
  (`features.manipulation`) — all modular, toggle-able, with config-driven
  look-back windows. Optional **safety gate** (`safety.enabled`) blocks or scales
  trades on suspicion scores (execution layer only).
- **Open weights by default** — versioned models export as plain LightGBM files +
  JSON metadata (`ModelRegistry.export_open_bundle`); no encryption or load gates.
- **Prediction + outcome logging** to SQLite (full feature vectors at prediction
  time; realised outcomes with context after the horizon) and tooling to rebuild
  training datasets from logged history.
- **Backtester** with proper trading metrics (Sharpe, Sortino, Calmar, profit factor,
  max drawdown, win rate) including realistic fees + slippage.
- **Runs fully offline.** Public exchange APIs are often geo-blocked from cloud/CI;
  when CCXT is unreachable a realistic, regime-switching **synthetic dataset** is
  generated so the entire pipeline is runnable anywhere.
- **Clear extension paths** — incremental learning (River), live WebSocket streaming
  (ccxt.pro), MLflow tracking, vectorbt cross-checks (all optional, lazy-imported).

## Project layout

```
epoch_ai/
├── config/            # Pydantic models + YAML loader
├── data/              # CCXT downloader, synthetic fallback, cleaning, websocket
├── features/          # modular feature groups + pipeline + target builder
├── models/            # LightGBM wrapper, base interface, versioned registry
├── logging_system/    # SQLite prediction/outcome store + joiner
├── learning/          # ★ progressive walk-forward engine, checkpoints, promotion
├── backtesting/       # backtester + trading metrics
├── execution/         # risk manager + paper trader (separate from prediction)
├── tracking/          # optional MLflow wrapper
├── utils/             # logging + timeframe helpers
└── cli.py             # download / backtest / paper-trade / info
config/config.yaml     # example config (progressive params highlighted)
tests/                 # pytest suite
```

## Quick start

epochAI has two primary workflows:

| Mode | Purpose | Command |
| --- | --- | --- |
| **Train** | Walk-forward learning, register versioned models | `python -m epoch_ai train` |
| **Run** | Load a trained model, predict + paper-execute | `python -m epoch_ai run` |

Future **Telegram** and **website** interfaces will call the same `TrainingService` and
`RuntimeService` in `epoch_ai/services/` (see `docs/adr/0003-train-run-interfaces.md`).

**New here?** Copy-paste the commands in [Train the model](#train-the-model) —
[first-time training](#3-first-time-training) then
[repeat to improve](#4-repeat-to-improve-the-model). The
[full pipeline](#quick-start-full-pipeline) section covers backtest, paper-trade,
tuning, and live replay.

## Train the model

Copy-paste commands below. Run them in order the first time, then repeat
[step 4](#4-repeat-to-improve-the-model) whenever you want to refresh data and retrain.

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-dev.txt   # ruff + pytest (for development)
```

On Windows, use `.venv\Scripts\python.exe` instead of `.venv/bin/python`. CCXT is
optional; with `data.use_synthetic_fallback: true` the pipeline runs fully offline when
the exchange is unreachable.

Optional: `pip install -r requirements-optional.txt` for live CCXT downloads, GPU
backends (`xgboost`), MLflow, etc.

### 2. Configure (optional)

Edit `config/config.yaml` (symbol, timeframe, walk-forward params), then check it:

```bash
python -m epoch_ai info
```

Override a setting for one run without editing YAML:

```bash
python -m epoch_ai train --set walk_forward.step_size=100
```

### 3. First-time training

Run these once to download history, train, and verify the model loads:

```bash
# 1. Download (or synthesize) market history
python -m epoch_ai download --bars 16000

# 2. Train — walk-forward over all history; registers model + optional SQLite logs
python -m epoch_ai train --bars 16000 --log-predictions

# 3. Smoke-test the registered model
python -m epoch_ai run --bars 6000 --live-bars 100 \
    --long-threshold 0.5 --short-threshold 0.5
```

**Fast smoke test** (fewer bars, capped steps):

```bash
python -m epoch_ai download --bars 8000
python -m epoch_ai train --bars 8000 --max-steps 12
```

**GPU training** (optional, NVIDIA + `pip install xgboost`):

```bash
python -m epoch_ai train --bars 16000 --log-predictions \
    --set model.backend=xgboost --set model.device=cuda
```

When training finishes you should see:

```text
=== Training complete ===
Symbol            : BTC/USDT
Model version     : v_20250626_123456
Walk-forward steps: 70
Final train rows  : 14,000
```

Models are saved under `artifacts/models/v_*/` (open weights + metadata). During long
walk-forward runs the registry **auto-prunes** to the newest versions (see
[Pause, resume, and registry cleanup](#pause-resume-and-registry-cleanup)).

| Flag | Purpose |
| --- | --- |
| `--bars N` | Cap history length |
| `--max-steps N` | Limit walk-forward iterations (quick demos) |
| `--log-predictions` | Log each OOS prediction + outcome to SQLite for retrain |
| `--no-register` | Walk forward without writing to the registry |
| `--no-resume` | Start at step 0 but leave any checkpoint file on disk |
| `--fresh` | Delete the walk-forward checkpoint and restart from step 0 |
| `--set key=value` | Dotted config override (repeatable) |

### Pause, resume, and registry cleanup

Full-history training can take hours or days. Walk-forward **checkpoints** let you stop
and continue without redoing completed steps.

**Pause:** press `Ctrl+C` after a `Step N | train=...` line finishes (or anytime — the
last *completed* step is saved). You get a short summary instead of a traceback:

```text
=== Training interrupted ===
Progress saved at step 78 (cutoff=17600).
Model checkpoint     : v_79
Checkpoint file        : artifacts/checkpoints/walk_forward_BTC-USDT.json
```

**Resume:** run the same `train` command again (resume is on by default):

```bash
python -m epoch_ai train --log-predictions --set model.device=cuda
```

Look for `Resuming walk-forward from step …` in the logs.

**Check progress** (no training, reads checkpoint + optional SQLite logs):

```bash
python -m epoch_ai progress
python -m epoch_ai progress --watch              # live-updating TUI (Ctrl+C to exit)
python -m epoch_ai progress --watch --interval 5 # refresh every 5 seconds
# or: python -m epoch_ai checkpoint status --watch
```

Shows completed/total steps, percent done, steps remaining, cutoff, checkpoint model,
registry version count, and logged OOS accuracy when `--log-predictions` was used.
Use `--refresh-rows` to recompute the resolved row count from cached parquet.

**Start over:** discard saved progress:

```bash
python -m epoch_ai train --fresh --log-predictions
```

**Migrating an old run** (trained before checkpoints existed): after stopping on a
completed step line, seed a checkpoint using the same config file as `train`:

```bash
python -m epoch_ai checkpoint seed --last-step 75
python -m epoch_ai train --log-predictions
```

Use the number from the last `Step N | …` log line; resume continues at step `N+1` with
model `v_{N+1}`.

**Registry disk usage:** each walk-forward step registers a new `v_*` directory. By
default only the **10 newest** versions are kept (`model.retain_versions: 10`). The
**champion** (`artifacts/models/current.json`), the **walk-forward checkpoint** model,
and the version just saved are never deleted even if they fall outside that window.
Disable pruning with `model.retain_versions: null`.

| Path | Purpose |
| --- | --- |
| `artifacts/checkpoints/walk_forward_<symbol>.json` | Resume pointer (step, cutoff, model) |
| `artifacts/models/v_*/` | Versioned open-weights snapshots |
| `artifacts/models/current.json` | Promoted champion for `run` / `auto-retrain` |

Config knobs (also in [Progressive learning parameters](#progressive-learning-parameters-configconfigyaml)):

```yaml
walk_forward:
  checkpoint_enabled: true     # save after each step (default)
  checkpoint_path: null        # null = per-symbol file under artifacts/checkpoints/

model:
  retain_versions: 10          # prune older v_* after each save; null = keep all
```

### 4. Repeat to improve the model

After the first train, run this block on a schedule (daily, weekly, etc.) to pull fresh
data, retrain, and paper-trade with the updated model:

```bash
# 1. Refresh cached history
python -m epoch_ai download --bars 16000

# 2. Retrain from logged predictions (needs prior runs with --log-predictions)
python -m epoch_ai retrain --min-new-samples 50

# 3. Run paper session and keep logging outcomes for the next retrain
python -m epoch_ai run --bars 6000 --live-bars 300 --log-predictions \
    --long-threshold 0.5 --short-threshold 0.5
```

**Alternative step 2 — full historical retrain** (walk-forward again on all data):

```bash
python -m epoch_ai train --bars 16000 --log-predictions
```

**Alternative step 2 — safe auto-update** (train challenger, promote only if better):

```bash
python -m epoch_ai auto-retrain
python -m epoch_ai auto-retrain --promote-policy   # also train/promote PPO when rl.enabled
```

**Coarse daily loop** (post-initial train; uses `adaptation.coarse_step_size`):

```bash
python -m epoch_ai schedule-retrain --promote --interval-hours 24 --max-cycles 1000
python -m epoch_ai schedule-retrain --promote --promote-policy --max-cycles 1  # smoke
```

**Holdout acceptance check** (predictor + policy vs baseline/buy-and-hold):

```bash
python -m epoch_ai evaluate-holdout --bars 8000
```

**Automate the repeat block** (retrain + promote on a timer):

```bash
python -m epoch_ai schedule-retrain --promote --interval-hours 24 --max-cycles 1000
```

**Inline retrain during a run** (refit every N bars without a separate job):

```bash
python -m epoch_ai run --bars 6000 --live-bars 300 --retrain-every 50 \
    --log-predictions --long-threshold 0.5 --short-threshold 0.5
```

**Export** the promoted model for sharing:

```bash
python -m epoch_ai export
```

| Command | Use when |
| --- | --- |
| `python -m epoch_ai download` | Refresh parquet cache before each retrain |
| `python -m epoch_ai train` | Full walk-forward retrain on all history |
| `python -m epoch_ai retrain` | Refit from SQLite logs after `--log-predictions` runs |
| `python -m epoch_ai auto-retrain` | Challenger vs champion; promote only if metric improves |
| `python -m epoch_ai schedule-retrain --promote` | Hands-off repeat of `auto-retrain` |
| `python -m epoch_ai run --log-predictions` | Paper/live session that feeds the next `retrain` |
| `python -m epoch_ai run --retrain-every N` | Retrain inside a replay session every N bars |

---

## Quick start (full pipeline)

The sections below cover download, run, backtest, paper-trade, tuning, and live
replay. For **training only**, use [Train the model](#train-the-model) above.

### 1. Download (or synthesize) the longest possible history

```bash
python -m epoch_ai download --bars 16000
```

If the exchange is reachable, CCXT fetches OHLCV (+ funding history) from
`historical_start_date` forward. Otherwise a synthetic regime-switching dataset is
generated and cached to `artifacts/data/`.

### 2. Train the AI

See [Train the model](#train-the-model) for the full step-by-step guide. Minimal
command:

```bash
python -m epoch_ai train --bars 16000 --log-predictions
```

### 3. Run the trained model on live data (predict + trade)

Simulated live feed (offline-safe — grows bar-by-bar like real streaming):

```bash
python -m epoch_ai run --live-feed --bars 6000 --live-bars 100 \
    --log-predictions --reserve-fraction 0.2 \
    --long-threshold 0.5 --short-threshold 0.5
```

Real exchange WebSocket (requires ccxt.pro + reachable exchange):

```bash
python -m epoch_ai run --live-stream --log-predictions
```

Real money (opt-in only — requires API keys in `EPOCH_AI_API_KEY` / `EPOCH_AI_API_SECRET`):

```bash
python -m epoch_ai run --live-stream --confirm-live --log-predictions
```

Session profits are split per `execution.reserve_fraction`: wins can be **reinvested**
(trading capital) or **set aside** (reserved wins in `artifacts/treasury.json`).

### 4. Run the trained model (paper replay batch mode)

```bash
python -m epoch_ai run --bars 6000 --live-bars 300 \
    --long-threshold 0.5 --short-threshold 0.5
```

Loads the latest registry model and steps bar-by-bar with risk rules + paper execution.
Requires a prior `train` (or `backtest --register-models`).

### 5. Run the first end-to-end progressive historical-learning backtest

```bash
python -m epoch_ai backtest --bars 16000 --log-predictions --register-models
```

This trains on the oldest data, walks forward through history (retraining each step),
logs every prediction + outcome to SQLite, versions each model, and prints a metrics
report plus the **out-of-sample learning curve** (accuracy first half vs second half).
Artifacts are written to `artifacts/backtests/` (`metrics.json`, `equity_curve.csv`,
`step_history.csv`, `feature_importance.csv`).

### 6. Simulate near-real-time paper trading with periodic updates

```bash
python -m epoch_ai paper-trade --bars 6000 --live-bars 300 \
    --long-threshold 0.5 --short-threshold 0.5
```

Trains on all-but-the-last `--live-bars` candles, then steps bar-by-bar through the
held-out tail predicting, applying risk rules and paper-executing (fees + slippage,
mark-to-market). The threshold overrides force directional positions on
hard-to-predict data so the execution path is exercised.

Use `--retrain-every N` for inline expanding-window retrains during the replay.

### 7. Hyperparameter sweep and config overrides

```bash
python -m epoch_ai tune --sweep config/sweeps/example.yaml --bars 4000 --max-steps 3
python -m epoch_ai backtest --set walk_forward.step_size=100 --max-steps 5
```

### 8. Periodic retrain from logs

```bash
python -m epoch_ai retrain --min-new-samples 50
```

### 8b. Automated, self-updating model (promote only if better)

```bash
# One-shot: train a challenger, score it + the current champion on a fresh holdout,
# and only repoint the live ("promoted") model when the challenger improves the metric.
python -m epoch_ai auto-retrain

# On a cadence (use an OS scheduler for robustness, or the built-in loop):
python -m epoch_ai schedule-retrain --promote --interval-hours 24 --max-cycles 1000
```

The registry tracks a **champion** pointer (`artifacts/models/current.json`); runtime
loads the promoted model and falls back to the latest version when none is set. The
challenger trains on data up to a holdout (minus the embargo gap), so the comparison is
genuinely out-of-sample. Tune the gate under `promotion:` in `config/config.yaml`
(`eval_bars`, `metric`, `min_improvement`).

### 9. Live replay (offline) or WebSocket stream

```bash
python -m epoch_ai live --replay --bars 6000 --live-bars 300
# python -m epoch_ai live   # requires ccxt.pro + reachable exchange
```

### Inspect the resolved configuration

```bash
python -m epoch_ai info
```

## Progressive learning parameters (`config/config.yaml`)

```yaml
walk_forward:
  initial_train_period: 2000   # candles for the first fit (oldest data)
  step_size: 200               # candles predicted + ingested per iteration
  retrain_frequency: 1         # retrain every N steps
  expanding: true              # expanding window = full accumulated history
  recency_half_life: 4000      # decay sample weights toward recent regimes (null = off)
  embargo: null                # purge gap between train/test (null = prediction.horizon; 0 = off)
  max_steps: null              # cap iterations for quick demos
  checkpoint_enabled: true     # persist resume state after each step
  checkpoint_path: null        # null = artifacts/checkpoints/walk_forward_<symbol>.json
```

The same engine powers both the **backtest simulation** and a **live retraining job**:
predict → collect outcomes + context → append samples → retrain → advance. See
[Repeat to improve the model](#4-repeat-to-improve-the-model) for the retrain commands.

The `embargo` gap purges the final bars of each training window so the forward-return
labels (computed over `prediction.horizon` bars) cannot overlap — and therefore leak —
the prediction window. It defaults to the horizon, which removes that look-ahead bias.

### Model, calibration and evaluation knobs

```yaml
prediction:
  neutral_band: 0.0            # >0 drops near-zero moves so labels reflect decisive moves

model:
  backend: lightgbm            # lightgbm (default) | xgboost (optional CUDA-GPU backend)
  val_fraction: 0.15           # time-ordered tail for early stopping + calibration
  class_weight: balanced       # derive scale_pos_weight from label balance (or "none")
  calibration: isotonic        # post-hoc P(up) calibration: isotonic | sigmoid | none
  refit_full_after_es: true    # refit on full window for the ES-chosen rounds (keep freshest bars)
  device: cpu                  # cpu | gpu | cuda — auto-falls back to cpu if unavailable
  gpu_platform_id: -1          # OpenCL platform id for LightGBM device=gpu (-1 = auto)
  gpu_device_id: -1            # OpenCL/CUDA device ordinal (-1 = auto)
  retain_versions: 10          # auto-delete oldest v_* after each save; null = keep all
  params:
    lambda_l1: 0.1             # mild regularisation (was 0.0)
    lambda_l2: 1.0

features:                      # look-back windows are config-driven
  return_lags: [1, 3, 6, 12, 24, 48]
  ma_windows: [10, 20, 50, 100, 200]
  rsi_periods: [7, 14, 28]
  vol_windows: [12, 24, 48, 96]
  sentiment: false             # joins a `fear_greed` column when present
  onchain: false               # joins on-chain columns (e.g. `exchange_netflow`)

backtest:
  horizon_aware: true          # hold each signal for prediction.horizon bars
```

**Model backends & GPU acceleration (optional).** The learner is pluggable via
`model.backend`:

- **`evolved_nn`** (default) — evolutionary PyTorch MLP on engineered features. Requires
  `pip install torch`. Training uses **real** exchange or cached parquet data (synthetic
  fallback is disabled). Tune search with `model.evolution.*` and `model.nn.*`.
- **`lightgbm`** — fast CPU GBM training. `model.device=gpu` uses LightGBM's OpenCL
  backend (requires a GPU-enabled LightGBM build).
- **`xgboost`** (optional, `pip install xgboost`) — ships prebuilt **CUDA wheels**, so
  `model.device=cuda` trains on NVIDIA GPUs out of the box. On large datasets this is a
  real speed-up (≈2× on ~1M rows in local benchmarks); on small/medium tabular data
  (≲200k rows) tree-boosting is CPU-friendly and GPU is roughly on par due to transfer
  overhead — so CPU stays the sensible default.

Either way, a GPU request that the installed build/host cannot satisfy logs a warning
and **automatically falls back to CPU**, so models can always be trained. Pin a device
on multi-GPU hosts with `gpu_device_id` (and `gpu_platform_id` for LightGBM OpenCL).
Both backends store **open weights** in the registry (`model.txt` for LightGBM,
`model.json` for XGBoost) and share the same calibration/early-stopping behaviour.

```bash
pip install xgboost                                  # one-time
python -m epoch_ai train --set model.backend=xgboost --set model.device=cuda
```

Per-step out-of-sample metrics (`step_history.csv`) now include `oos_logloss`,
`oos_brier`, `oos_auc`, `oos_directional_accuracy` and `oos_coverage` so the learning
curve reflects the decision the system actually trades.

## Development

```bash
ruff check .                 # lint
pytest                       # tests
pre-commit install           # optional local hooks (ruff + pytest)
```

CI runs **ruff** and **pytest** on every push to `master` (`.github/workflows/ci.yml`).

Backtest artifacts now include `learning_curve.json` (rolling OOS accuracy, trend slope).

Architecture decisions: `docs/adr/`.

## Tech stack

Python 3.12+ · pandas/numpy · PyTorch (evolved_nn + PPO policy) · LightGBM · Pydantic + YAML · SQLite + Parquet ·
scikit-learn. Optional: ccxt, vectorbt, mlflow, river, pandas_ta, python-telegram-bot.

## Disclaimer

This is **research and educational software only**. It is **not** financial advice,
investment advice, or a recommendation to trade. Paper trading is the default;
real-money exchange order routing is **intentionally unimplemented** (`execution.mode`
stays `paper` unless you explicitly enable live keys and `--confirm-live` — use at your
own risk). Backtested, synthetic, and paper results do **not** imply live profitability.
Multi-horizon forecasts and learned policies can fail on unseen regimes; hard caps and
kill-switches are safety nets, not guarantees.

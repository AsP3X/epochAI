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
- **Calibrated, class-balanced model** — LightGBM with balanced class weighting,
  post-hoc probability calibration (isotonic/Platt) fit on a held-out validation
  tail, and mild L1/L2 regularisation, so `P(up)` is trustworthy for thresholding.
- **Honest, horizon-aware evaluation** — per-step OOS metrics include logloss,
  Brier, ROC-AUC and execution-threshold-aware accuracy; the backtest holds each
  signal for the full prediction horizon (`backtest.horizon_aware`).
- **Rich, multi-source features** — technical (incl. ADX, VWAP, OBV, CCI,
  Williams %R), microstructure, derivatives (funding / open interest /
  liquidations), volatility/regime, cyclical time, plus optional sentiment
  (Fear & Greed) and on-chain (exchange net-flow) groups — all modular,
  toggle-able, with config-driven look-back windows.
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
├── learning/          # ★ progressive walk-forward engine (the core)
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

```bash
python3 -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows (PowerShell):
#   .venv\Scripts\Activate.ps1
pip install -r requirements.txt          # core
pip install -r requirements-dev.txt      # ruff + pytest
# optional integrations (ccxt, vectorbt, mlflow, river, pandas_ta):
# pip install -r requirements-optional.txt
```

On Windows, use `.venv\Scripts\python.exe` instead of `.venv/bin/python` for lint and
tests (e.g. `.venv\Scripts\python.exe -m pytest`). CCXT is optional; with
`data.use_synthetic_fallback: true` (the default) the pipeline runs fully offline.

### 1. Download (or synthesize) the longest possible history

```bash
python -m epoch_ai download --bars 16000
```

If the exchange is reachable, CCXT fetches OHLCV (+ funding history) from
`historical_start_date` forward. Otherwise a synthetic regime-switching dataset is
generated and cached to `artifacts/data/`.

### 2. Train the AI (progressive walk-forward + model registry)

```bash
python -m epoch_ai train --bars 16000 --log-predictions
```

This walks forward through history, learns from realised outcomes, and saves versioned
models under `artifacts/models/`.

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

### 4. Run the first end-to-end progressive historical-learning backtest

```bash
python -m epoch_ai backtest --bars 16000 --log-predictions --register-models
```

This trains on the oldest data, walks forward through history (retraining each step),
logs every prediction + outcome to SQLite, versions each model, and prints a metrics
report plus the **out-of-sample learning curve** (accuracy first half vs second half).
Artifacts are written to `artifacts/backtests/` (`metrics.json`, `equity_curve.csv`,
`step_history.csv`, `feature_importance.csv`).

### 5. Simulate near-real-time paper trading with periodic updates

```bash
python -m epoch_ai paper-trade --bars 6000 --live-bars 300 \
    --long-threshold 0.5 --short-threshold 0.5
```

Trains on all-but-the-last `--live-bars` candles, then steps bar-by-bar through the
held-out tail predicting, applying risk rules and paper-executing (fees + slippage,
mark-to-market). The threshold overrides force directional positions on
hard-to-predict data so the execution path is exercised.

Use `--retrain-every N` for inline expanding-window retrains during the replay.

### 6. Hyperparameter sweep and config overrides

```bash
python -m epoch_ai tune --sweep config/sweeps/example.yaml --bars 4000 --max-steps 3
python -m epoch_ai backtest --set walk_forward.step_size=100 --max-steps 5
```

### 7. Periodic retrain from logs

```bash
python -m epoch_ai retrain --min-new-samples 50
```

### 8. Live replay (offline) or WebSocket stream

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
  max_steps: null              # cap iterations for quick demos
```

The same engine powers both the **backtest simulation** and a **live retraining job**:
predict → collect outcomes + context → append samples → retrain → advance.

### Model, calibration and evaluation knobs

```yaml
model:
  val_fraction: 0.15           # time-ordered tail for early stopping + calibration
  class_weight: balanced       # derive scale_pos_weight from label balance (or "none")
  calibration: isotonic        # post-hoc P(up) calibration: isotonic | sigmoid | none
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

Python 3.12+ · pandas/numpy · LightGBM · Pydantic + YAML · SQLite + Parquet ·
scikit-learn. Optional: ccxt, vectorbt, mlflow, river, pandas_ta.

## Disclaimer

This is research/educational software. It is **not** financial advice and ships with
no profitability guarantees. Backtested/synthetic results do not imply live returns.

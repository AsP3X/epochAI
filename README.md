# epochAI

A self-contained, self-improving **crypto AI trading prediction system** built around
**progressive (expanding-window) walk-forward learning**. The model starts on the
*oldest* available history for a symbol, predicts the next unseen period, ingests the
realised outcomes plus rich influencing context, retrains, and marches forward through
time — building deep, long-term understanding across many market regimes before
operating closer to real-time.

> Prediction is kept strictly separate from execution/risk management.

---

## Highlights

- **Progressive historical learning engine** — expanding/rolling walk-forward with
  configurable retraining frequency and optional recency weighting.
- **Rich, multi-source features** — technical, microstructure, derivatives
  (funding / open interest / liquidations), volatility/regime, and cyclical time
  features, all modular and toggle-able.
- **LightGBM** model wrapper with time-ordered early stopping, feature importance,
  saving/loading and a file-based **versioned registry**.
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

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # core
pip install -r requirements-dev.txt      # ruff + pytest
# optional integrations (ccxt, vectorbt, mlflow, river, pandas_ta):
# pip install -r requirements-optional.txt
```

### 1. Download (or synthesize) the longest possible history

```bash
python -m epoch_ai download --bars 16000
```

If the exchange is reachable, CCXT fetches OHLCV (+ funding history) from
`historical_start_date` forward. Otherwise a synthetic regime-switching dataset is
generated and cached to `artifacts/data/`.

### 2. Run the first end-to-end progressive historical-learning backtest

```bash
python -m epoch_ai backtest --bars 16000 --log-predictions --register-models
```

This trains on the oldest data, walks forward through history (retraining each step),
logs every prediction + outcome to SQLite, versions each model, and prints a metrics
report plus the **out-of-sample learning curve** (accuracy first half vs second half).
Artifacts are written to `artifacts/backtests/` (`metrics.json`, `equity_curve.csv`,
`step_history.csv`, `feature_importance.csv`).

### 3. Simulate near-real-time paper trading with periodic updates

```bash
python -m epoch_ai paper-trade --bars 6000 --live-bars 300 \
    --long-threshold 0.5 --short-threshold 0.5
```

Trains on all-but-the-last `--live-bars` candles, then steps bar-by-bar through the
held-out tail predicting, applying risk rules and paper-executing (fees + slippage,
mark-to-market). The threshold overrides force directional positions on
hard-to-predict data so the execution path is exercised.

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
  recency_half_life: null      # e.g. 5000 to emphasise recent regimes
  max_steps: null              # cap iterations for quick demos
```

The same engine powers both the **backtest simulation** and a **live retraining job**:
predict → collect outcomes + context → append samples → retrain → advance.

## Development

```bash
ruff check .                 # lint
pytest                       # tests
```

## Tech stack

Python 3.11+ · pandas/numpy · LightGBM · Pydantic + YAML · SQLite + Parquet ·
scikit-learn. Optional: ccxt, vectorbt, mlflow, river, pandas_ta.

## Disclaimer

This is research/educational software. It is **not** financial advice and ships with
no profitability guarantees. Backtested/synthetic results do not imply live returns.

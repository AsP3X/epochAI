# epochAI Operator Runbook

This runbook covers day-to-day operations for training, running, monitoring, and exporting models.

## Prerequisites

```bash
pip install -r requirements.txt -r requirements-dev.txt
# Optional: API, Telegram, exchange, MLflow
pip install -r requirements-optional.txt
```

## Train a model

```bash
python -m epoch_ai train --bars 5000 --log-predictions
```

Models are versioned under `artifacts/models/v_*`.

## Run (paper / simulated live feed)

```bash
python -m epoch_ai run --live-feed --bars 5000 --live-bars 100 --log-predictions
```

Use `--reserve-fraction 0.2` to set aside 20% of session wins. Cold storage and daily profit caps are configured in `config/config.yaml` under `execution`.

## Live exchange (dry-run by default)

1. Set API keys: `EPOCH_AI_API_KEY`, `EPOCH_AI_API_SECRET`
2. Enable in config: `execution.live_enabled: true`
3. Run with `--confirm-live` (still respects kill switch and calibration gates)

## Kill switch

Halt all rebalancing immediately:

```bash
python -m epoch_ai kill-switch halt --reason "maintenance window"
python -m epoch_ai kill-switch status
python -m epoch_ai kill-switch resume
```

The kill switch file defaults to `artifacts/kill_switch.json` and is shared by CLI, API, and Telegram.

## HTTP API

```bash
python -m epoch_ai serve --host 0.0.0.0 --port 8000
```

Endpoints:

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Readiness + issues |
| GET | `/status` | Model registry summary |
| GET | `/models` | List registered versions |
| POST | `/predict/latest` | Predict on latest cached/historical bars |
| POST | `/train` | Trigger training |
| POST | `/export` | Export open-weights bundle |
| POST | `/kill/halt` | Halt trading |
| POST | `/kill/resume` | Resume trading |

## Export open weights

```bash
python -m epoch_ai export --dest artifacts/exports --model-version v_1
```

Produces `model.txt`, `metadata.json`, `README.txt`, and `MODEL_CARD.md`. No license is bundled — see repository owner.

## Tune and promote

```bash
python -m epoch_ai tune --sweep config/sweeps/example.yaml --out artifacts/sweeps
python -m epoch_ai promote --sweep-out artifacts/sweeps --out config/promoted.yaml
```

## Periodic retrain

```bash
python -m epoch_ai retrain --min-new-samples 50
python -m epoch_ai schedule-retrain --interval-hours 24 --max-cycles 1
```

## Telegram bot (optional)

Set `EPOCH_AI_TELEGRAM_TOKEN` and optionally `telegram.allowed_chat_ids` in config.

```bash
python -m epoch_ai telegram
```

Commands: `/status`, `/predict`, `/halt [reason]`, `/resume`

## Monitoring artifacts

| Path | Contents |
|------|----------|
| `artifacts/audit/trades.jsonl` | Prediction, fill, halt events |
| `artifacts/metrics/runtime.jsonl` | Per-tick equity and signal snapshots |
| `artifacts/logs/predictions.sqlite` | Predictions + outcomes for retrain |
| `artifacts/treasury.json` | Trading capital, reserved, cold storage |

## Calibration gate

When `execution.calibration_min_accuracy` is set, live rebalancing is blocked if rolling out-of-sample accuracy (from resolved predictions) falls below the threshold after `calibration_min_samples` outcomes.

## Troubleshooting

- **No models in registry** — run `train` first.
- **Warmup not complete** — increase `--bars` or lower `execution.min_buffer_bars`.
- **Kill switch active** — run `kill-switch resume` or delete `artifacts/kill_switch.json`.
- **CCXT geo-block** — synthetic fallback is used automatically when `data.use_synthetic_fallback: true`.

## CI / release

- CI runs ruff + pytest on every push to `master`.
- Tag releases (`v*`) trigger `.github/workflows/release.yml` to build sdist/wheel artifacts.

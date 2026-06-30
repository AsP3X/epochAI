# epochAI Operator Runbook

This runbook covers day-to-day operations for training, running, monitoring, and exporting
models. For a first-time walkthrough see **`docs/get-started.md`**.

## Prerequisites

```bash
pip install -r requirements.txt -r requirements-dev.txt
# Optional: API, Telegram, exchange, MLflow
pip install -r requirements-optional.txt
```

## Train a model

**Requires real exchange data** (provenanced parquet). Download first:

```bash
python -m epoch_ai download --full-history
python -m epoch_ai train --set model.device=cuda
python -m epoch_ai evaluate-holdout
```

Models are versioned under `artifacts/models/v_*`. Long runs **auto-save a resume
checkpoint** after each walk-forward step and **prune** older versions (default: keep the
10 newest — see `model.retain_versions` in `config/config.yaml`).

### Pause and resume

1. Stop with `Ctrl+C` (preferably right after a `Step N | …` log line).
2. Resume with the same command — no flags needed:

   ```bash
   python -m epoch_ai train --set model.device=cuda
   ```

3. Restart from step 0:

   ```bash
   python -m epoch_ai train --fresh --set model.device=cuda
   ```

Checkpoint file (default): `artifacts/checkpoints/walk_forward_BTC-USDT.json`.

**Legacy run without checkpoints:** seed from the last completed log step, then resume:

```bash
python -m epoch_ai checkpoint seed --last-step 75
python -m epoch_ai train --set model.device=cuda
```

Use `config/config.yaml` (or the same `--config` / `--set` overrides as `train`) when
seeding — the fingerprint must match the training config (feature count, walk-forward
params).

### Check progress

Without starting training:

```bash
python -m epoch_ai progress
python -m epoch_ai progress --watch --interval 2
# alias:
python -m epoch_ai checkpoint status --watch
```

Reports completed/total walk-forward steps, percent done, steps remaining, checkpoint
cutoff/model, registry version count, and SQLite OOS accuracy when predictions were logged.
Add `--refresh-rows` to recompute resolved row count from cached parquet.

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
| `artifacts/models/v_*/` | Versioned open-weights models (auto-pruned during train) |
| `artifacts/models/current.json` | Promoted champion pointer |
| `artifacts/checkpoints/walk_forward_*.json` | Walk-forward resume state |
| `artifacts/audit/trades.jsonl` | Prediction, fill, halt events |
| `artifacts/metrics/runtime.jsonl` | Per-tick equity and signal snapshots |
| `artifacts/logs/predictions.sqlite` | Predictions + outcomes for retrain |
| `artifacts/treasury.json` | Trading capital, reserved, cold storage |

## Calibration gate

When `execution.calibration_min_accuracy` is set, live rebalancing is blocked if rolling out-of-sample accuracy (from resolved predictions) falls below the threshold after `calibration_min_samples` outcomes.

## Troubleshooting

- **No models in registry** — run `train` first.
- **Checkpoint config mismatch on resume** — seed with `checkpoint seed` using the same
  `config/config.yaml` as `train`, or run `train --fresh`.
- **Warmup not complete** — increase `--bars` or lower `execution.min_buffer_bars`.
- **Kill switch active** — run `kill-switch resume` or delete `artifacts/kill_switch.json`.
- **No provenance / synthetic cache rejected** — `python -m epoch_ai download --full-history --force`
- **CCXT geo-block** — `train` fails without real data; use `backtest` with
  `--set data.use_synthetic_fallback=true` for offline CI smokes only.

## CI / release

- CI runs ruff + pytest on every push to `master`.
- Tag releases (`v*`) trigger `.github/workflows/release.yml` to build sdist/wheel artifacts.

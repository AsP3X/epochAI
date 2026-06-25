# ADR 0003: Train mode vs run mode and future interfaces

## Status

Accepted

## Context

epochAI must support two distinct operator workflows:

1. **Training** — walk forward through history, learn from outcomes, version models.
2. **Running** — load a trained model, predict on new bars, apply risk rules, paper-execute.

Later we will add **Telegram** and **website** interfaces. Those must not reimplement
CLI logic or reach into engines directly.

## Decision

Introduce a **service layer** under `epoch_ai/services/`:

| Service | Mode | Responsibility |
| --- | --- | --- |
| `TrainingService` | Train | `train()`, `backtest()`, `retrain()`, `tune()`, `list_models()` |
| `RuntimeService` | Run | `load_model()`, `predict_market()`, `run_session()`, `status()` |

CLI exposes first-class commands:

- `python -m epoch_ai train` — train + register model
- `python -m epoch_ai run` — load registry model + paper/replay session

Existing commands (`backtest`, `paper-trade`, `retrain`, …) remain for power users;
new interfaces should prefer the services.

Models are persisted in the file-based `ModelRegistry` (`artifacts/models/v_*`).
Runtime **requires** a registered model unless inline retrain is explicitly enabled.

## Consequences

- Telegram bot and website API can call the same Python services (later: thin HTTP
  wrapper or task queue) without duplicating walk-forward or live-loop code.
- Clear separation helps ops: train on schedule / on demand, run continuously with
  frozen or periodically retrained weights.
- `RuntimeService.status()` provides a stable health snapshot for dashboards and bots.

## Future work (out of scope today)

- HTTP API (`FastAPI`) wrapping `TrainingService` / `RuntimeService`
- Telegram bot commands: `/train`, `/predict`, `/status`, `/run`
- Website dashboard for learning curves, model versions, and paper PnL
- Authentication and multi-user model isolation

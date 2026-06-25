# ADR 0002: Prediction vs execution separation

## Status

Accepted

## Context

Mixing model training with position sizing and order logic makes backtests hard to
interpret and couples research code to brokerage constraints.

## Decision

Keep **prediction** (`learning/`, `features/`, `models/`, `logging_system/`) separate
from **execution** (`execution/` risk + paper trader) and **simulation metrics**
(`backtesting/`). The progressive engine emits probabilities; `RiskManager` applies
thresholds, halts, and sizing; `PaperTrader` simulates fills.

## Consequences

- Threshold and risk changes do not require retraining.
- Live/replay loops share `execution/live_loop.py` without touching LightGBM code.
- Cross-boundary changes need explicit justification and tests on both sides.

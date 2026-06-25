# ADR 0004: Live data, predictions, and trade execution

## Status

Accepted

## Context

After training, the finished model must:

1. Receive **live data** from exchange/data sources.
2. Produce **predictions** on price (course) for each new candle.
3. **Initialize trades** to capture gains, with profits **reinvested** or **set aside**.

Telegram and website interfaces will orchestrate this path later.

## Decision

### Pipeline

```
Live OHLCV feed (WebSocket or simulated feed)
    -> LiveTradingEngine.process_bar()
        -> RuntimeService.predict_market()   # trained registry model
        -> RiskManager.decide()              # thresholds, halts
        -> TradeExecutor.rebalance()         # paper or live exchange
        -> PredictionStore (optional)        # for future retraining
    -> Treasury.allocate_session_pnl()     # reinvest vs reserve wins
```

### Components

| Module | Role |
| --- | --- |
| `RealtimeDataHandler` | Live candle buffer from ccxt.pro |
| `LiveTradingEngine` | Orchestrates predict → trade → outcome logging |
| `TradeExecutor` | `PaperExecutor` (default) or `LiveExecutor` (ccxt orders) |
| `Treasury` | Splits session PnL: `reserve_fraction` set aside, rest reinvested |

### CLI

```bash
python -m epoch_ai train --bars 16000
python -m epoch_ai run --live-feed --log-predictions --reserve-fraction 0.2
python -m epoch_ai run --live-stream --confirm-live   # real orders + API keys
```

Live exchange orders require `execution.live_enabled`, API env vars, and `--confirm-live`.
Default is **paper/dry-run** so the pipeline is testable without capital at risk.

## Consequences

- Clear path from trained model to live predictions and trades.
- Outcomes logged to SQLite enable periodic `retrain` on live performance.
- Treasury state persists in `artifacts/treasury.json`.
- Real-money trading remains opt-in; research disclaimer still applies.

## Future work

- Telegram alerts on fills and treasury updates
- Website dashboard for live PnL and reserved wins
- Order types beyond market rebalance (limits, stops)
- Multi-symbol portfolio execution

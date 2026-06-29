# ADR 0008: Multi-Horizon Prediction and Learned Trading Policy

## Status

Accepted (2026-06-29)

## Context

epochAI previously predicted a single forward direction over a fixed horizon (15m base,
12-candle / 4h target). The product goal is to forecast price at multiple near-term
horizons (1m through 1hr) with honest uncertainty bands, expose those forecasts to a
candle chart, and drive an automatic paper-trading bot that learns entry, exit, and size.

This requires:

1. A **1-minute base timeframe** so sub-15m horizons are meaningful.
2. A **multi-head neural predictor** emitting quantile return bands and calibrated P(up)
   per horizon from one shared trunk.
3. A **learned RL policy** (PPO) that consumes predictor outputs + features to decide
   trades — crossing the usual prediction/execution separation documented in ADR 0002.

The repository owner explicitly authorized this boundary crossing for v1 (paper-only).

## Decision

- **Prediction layer:** `PredictionConfig.horizons` + multi-head `evolved_nn`; canonical
  artifact `MultiHorizonPredictionResult` (planned Phase 4).
- **Execution layer:** RL policy isolated under `epoch_ai/execution/policy/` with hard
  caps (leverage, drawdown kill-switch, max-hold). Baseline ensemble policy retained as
  benchmark/fallback.
- **Evaluation:** final holdout must beat buy-and-hold and baseline net of fees/funding.
- **Feedback loop:** structured action log feeds future retraining.

Implementation plan: `docs/superpowers/plans/2026-06-29-multi-horizon-prediction.md`.

## Consequences

- Fresh train required; existing single-head checkpoints are incompatible.
- Tests must cover both prediction and policy layers.
- Real 1m microstructure data is required for short-horizon edge; synthetic is
  plumbing-only.
- Open-weights policy: predictor bundle + RL policy are plain publishable files.

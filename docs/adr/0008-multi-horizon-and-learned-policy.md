# ADR 0008: Multi-Horizon Prediction and Learned Trading Policy

## Status

Accepted (2026-06-29) — implemented through Phase 7.

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
  artifact `MultiHorizonPredictionResult` in `epoch_ai/services/types.py`.
- **Execution layer:** RL policy isolated under `epoch_ai/execution/policy/` with hard
  caps (leverage, drawdown kill-switch, max-hold). Baseline ensemble policy retained as
  benchmark/fallback (`execution/policy/baseline.py`).
- **Evaluation:** final holdout (`adaptation.holdout_bars` / `promotion.eval_bars`) is
  never trained on; `evaluate-holdout` scores predictor Brier and policy vs baseline +
  buy-and-hold net of fees.
- **Feedback loop:** `execution/action_log.py` JSONL feeds `run_retrain` sample-weight
  boosts; policy cycles use `learning/policy_promotion.py` with benchmark gates.
- **Coarse adaptation:** after the initial full walk-forward train, `schedule-retrain
  --promote` applies `adaptation.coarse_step_size` / daily cadence.

Implementation plan: `docs/superpowers/plans/2026-06-29-multi-horizon-prediction.md`.

## Boundary isolation

| Layer | Modules | Must not contain |
| --- | --- | --- |
| Prediction | `features/`, `models/`, `learning/progressive.py` | position sizing, PPO, fees |
| Policy | `execution/policy/` | training labels from simulated PnL (unless explicit) |
| Orchestration | `services/`, `cli.py` | model internals |

The supervised predictor remains independently trainable (`train`, `predict --json`).
The PPO policy is optional (`rl.enabled`, `train-policy`, `run --policy learned`).

## Consequences

- Fresh train required; existing single-head checkpoints are incompatible.
- Tests cover both prediction and policy layers (`test_multi_head`, `test_policy_env`, …).
- Real 1m microstructure data is required for short-horizon edge; synthetic is
  plumbing-only.
- Open-weights policy: predictor bundle + RL policy are plain publishable files.
- Real-money order routing remains intentionally unimplemented; paper-only default.

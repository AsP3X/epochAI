# ADR 0001: Progressive expanding-window walk-forward learning

## Status

Accepted

## Context

Crypto markets are non-stationary. A single train/test split hides regime change and
overstates performance. epochAI must measure whether the model improves as it walks
forward through history.

## Decision

Use an **expanding-window walk-forward** loop in `epoch_ai/learning/progressive.py`:

1. Train on the oldest `initial_train_period` bars.
2. Predict the next `step_size` unseen bars (strictly out-of-sample).
3. Log predictions, outcomes, and context.
4. Retrain every `retrain_frequency` steps on accumulated history.
5. Advance until history is exhausted or `max_steps` is reached.

## Consequences

- Honest OOS metrics per step (`step_history`, learning curve artifacts).
- Runtime scales with steps × retrains — use `--max-steps` for smokes.
- All features must remain **causal** (`ml-causality.mdc`).

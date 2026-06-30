# TCN temporal backend + 5m migration — design

Status: implemented (2026-06-30). ruff clean; 306 tests pass (incl. TCN unit +
walk-forward); `python -m epoch_ai info` resolves at 5m/tcn.

## Goal

Add a causal **Temporal Convolutional Network (TCN)** prediction backend
(`model.backend: tcn`) that learns temporal structure directly from a sliding window
of the existing engineered feature rows, and migrate the production config from a 1m
to a 5m base timeframe. `evolved_nn` remains fully supported; TCN is additive.

Honest framing: this raises the model ceiling and the quality of OOS evaluation. It
does **not** guarantee profit — 5m BTC is noisy and a strong model wins ~52–56% of the
time. The promotion gate ensures only genuinely-better models go live.

## Decisions (locked)

- **Architecture:** dilated causal TCN (residual Conv1d blocks, weight-norm, dropout).
- **Input:** window of the last `L` engineered + scaled feature rows (reuses the whole
  causal feature pipeline + scaler). `L = model.tcn.lookback` (default 96 bars = 8h@5m).
- **Horizons (5m):** `[1, 3, 6, 12, 24, 48]` = 5m/15m/30m/1h/2h/4h; primary `12` (1h).
- **Migration:** rescale time-based duration knobs ÷5 to preserve wall-clock; keep
  feature lookback windows bar-based.

## Architecture & components

- `epoch_ai/models/base.py`: new `MultiHeadModel(BaseModel)` base exposing
  `multi_head_spec_`, `primary_horizon_`, `sequence_lookback` (None for dense models),
  `predict_logits`, `predict_structured`, `seed_payload()`. Both `EvolvedNNModel` and
  `TCNModel` subclass it. Engine/runtime/promotion/acceptance/live check the base
  instead of `EvolvedNNModel`.
- `epoch_ai/models/tcn_model.py`: `TCNModel(MultiHeadModel)`, `BACKEND="tcn"`,
  `MODEL_FILENAME="model.pt"`. Self-contained training loop reusing `multi_head.py`
  losses/parse, `calibration.py`, and device/CUDA helpers from `nn_trainer.py`.
- `epoch_ai/config/settings.py`: `TCNConfig` block under `ModelConfig.tcn`; `tcn` added
  to `backend` Literal and `factory.BACKENDS`.

## Causal windowing (critical)

- Windows are built **on the fly per batch** from the scaled 2D matrix kept on device:
  for batch indices `i`, gather rows `[i-L+1 … i]`; positions `< 0` are zero-padded
  (post-scaling mean ≈ 0). Never materialize the full `(n, L, F)` tensor (would be tens
  of GB on full history).
- A window for bar `t` uses only rows `≤ t` → no leakage. The embargo gap rows between
  train_end and the prediction cutoff are valid *input context* (they are past relative
  to the predicted bar) even though they are excluded from training labels.
- At prediction time a sequence model needs the `L-1` feature rows preceding the block.
  `sequence_lookback` lets callers pass a lookback context tail and trim:
  - `progressive.py` (OOS metrics): pass `x_all.iloc[cutoff-(L-1):test_end]`, trim
    leading rows from `predict`/`predict_structured` outputs.
  - `runtime`/`live_loop`: pass a feature tail instead of a single row.
  - `promotion`/`acceptance`: contiguous holdouts; internal zero-pad on the first L-1
    rows only (≈1.5% of eval bars) — documented, negligible.

## Config additions (`model.tcn`)

`lookback: 96`, `channels: [64,64,128,128]`, `kernel_size: 3`, `dropout: 0.1`,
`max_epochs`, `batch_size`, `patience`, `learning_rate`. Reuses shared
`model.calibration/class_weight/val_fraction/device/cuda/refit_full_after_es`.

## 5m migration (`config/config.yaml`)

`timeframe: 5m`; `horizons: [1,3,6,12,24,48]`, `horizon: 12`, `neutral_band: 0.0005`.
Rescale ÷5: `initial_train_period 8640`, `step_size 288`, `recency_half_life 15000`,
`promotion.eval_bars 6048`, `adaptation.coarse_step_size 864`, `trading.max_hold_bars
288`, `execution.min_buffer_bars 1500` (≥ lookback + max_horizon + feature warmup).

## Testing

`tests/test_tcn_model.py`: fit/predict/save/load round-trip, multi-head output shapes,
**causality test** (future rows cannot change a past prediction), context-tail
equivalence. Extend `test_progressive.py` (tiny walk-forward with `backend: tcn`) and
`test_config.py` (`model.tcn` validation). `pytest.importorskip("torch")`.

## Out of scope (separate follow-up)

Feature pruning (#4) via permutation importance — its own spec/plan after TCN baseline.

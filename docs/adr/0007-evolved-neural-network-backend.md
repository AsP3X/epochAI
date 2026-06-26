# ADR 0007: Evolved neural network prediction backend

## Status

Accepted

## Context

epochAI previously predicted short-horizon direction with gradient-boosted trees over
causal engineered features. Users requested an **evolutionary neural network** that
searches MLP architecture while preserving walk-forward honesty, open weights, and
real-market training data (not synthetic regimes for production training).

## Decision

1. **Default backend** becomes `model.backend=evolved_nn`: a PyTorch MLP whose layer
   sizes, dropout, and optimizer hyper-parameters are searched by a lightweight
   (μ+λ) evolutionary loop; each candidate is trained with Adam and early stopping
   on a time-ordered validation tail.

2. **Keep engineered features** (Option A). The NN consumes the existing causal
   `FeaturePipeline` output — no end-to-end raw-OHLCV CNN in this ADR.

3. **Real data for training.** `TrainingService` disables synthetic fallback when
   `evolved_nn` is active. If CCXT extension fails, **cached real parquet** is used
   (user-approved policy A).

4. **Fallback backends** remain: `lightgbm` and `xgboost` for fast CI and comparison.

5. **Open weights.** Registry stores `model.pt`, `model.pt.genome.json`,
   `model.pt.scaler.json`, and optional calibration sidecar — plain files, no DRM.

6. **Feature importance** uses permutation loss increase on the validation tail (NNs
   lack native tree gain).

## Consequences

**Positive**

- Architecture adapts per walk-forward retrain without hand-tuning tree depth/leaves.
- Same progressive engine, logging, calibration, and promotion gate semantics.
- PyTorch is lazy-optional; GBM backends still run without it.

**Negative**

- Training is slower than LightGBM (evolution × Adam epochs × walk-forward retrains).
- Quality gains require **real** historical data volume; synthetic pytest fixtures
  only validate plumbing, not alpha.

**Follow-up**

- Hybrid sequence encoder (Option C) if evolved MLP beats GBM on real-data holdouts.
- GPU-default training when CUDA is present.

## Non-goals

- Pure neuroevolution without gradient descent.
- RL training on simulated PnL as the primary objective.
- Replacing the feature pipeline with learned representations in v1.

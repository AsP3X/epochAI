# Evolved Neural Network Backend — Design Spec

**Date:** 2026-06-26  
**Status:** Approved (implementation in progress)

## Summary

Replace the default prediction backend with an **evolutionary PyTorch MLP** trained on
existing causal engineered features (Option A). Walk-forward evaluation, calibration,
and open-weights registry semantics are unchanged.

## Requirements

1. **Backend:** `model.backend=evolved_nn` (new default).
2. **Evolution:** μ+λ search over MLP architecture genes; Adam + early stopping per candidate.
3. **Fitness:** Validation logloss (classification) on time-ordered holdout tail.
4. **Real data for training:** `evolved_nn` disables synthetic fallback; use cached real
   parquet when CCXT extension fails (policy A).
5. **Fallbacks:** Keep `lightgbm` / `xgboost` for fast CI and comparison.
6. **Tests:** Synthetic `market` fixture for pipeline tests; optional real-data integration
   marked separately.

## Artifacts

| File | Role |
|------|------|
| `model.pt` | PyTorch state_dict + metadata |
| `model.pt.genome.json` | Evolved architecture |
| `model.pt.scaler.json` | StandardScaler params |
| `model.pt.calibration.json` | Optional probability calibrator |

## ADR

See `docs/adr/0007-evolved-neural-network-backend.md`.

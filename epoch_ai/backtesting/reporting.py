"""Formatting helpers for backtest and training reports."""

from __future__ import annotations

import pandas as pd

# Human: GBM backends expose tree gain; evolved_nn uses permutation drop in logloss.
# Agent: READS model.backend; RETURNS display label matching the underlying metric.
_IMPORTANCE_LABELS: dict[str, tuple[str, str]] = {
    "evolved_nn": ("permutation", "permutation importance"),
    "lightgbm": ("gain", "gain"),
    "xgboost": ("gain", "gain"),
}


def importance_metric_label(backend: str) -> str:
    """Return the user-facing name for feature-importance values."""
    return _IMPORTANCE_LABELS.get(backend, ("gain", "gain"))[1]


def format_importance_value(value: float) -> str:
    """Format a single importance score without rounding small values to zero."""
    if value <= 0.0:
        return "0"
    if value >= 100.0:
        return f"{value:.1f}"
    if value >= 1.0:
        return f"{value:.3f}"
    if value >= 0.01:
        return f"{value:.4f}"
    return f"{value:.2e}"


def count_rebalances(
    weights: pd.Series,
    *,
    horizon_aware: bool,
    horizon: int,
) -> int:
    """Count non-trivial position-weight changes (post horizon-aware smoothing)."""
    w = weights.astype(float)
    if horizon_aware:
        w = w.rolling(max(1, horizon), min_periods=1).mean()
    delta = w.diff().abs().fillna(w.abs())
    return int((delta > 1e-12).sum())

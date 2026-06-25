"""Per-step out-of-sample metrics for the progressive walk-forward engine.

These quantify genuine OOS quality for each walk-forward step. Classification adds
probabilistic (logloss, Brier, AUC) and execution-threshold-aware diagnostics so the
learning curve reflects the decision the system actually trades, not just a fixed-0.5
cut. Regression reports directional accuracy and RMSE so a regression task no longer
emits meaningless zero metrics.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-6


def classification_step_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    *,
    long_threshold: float,
    short_threshold: float,
) -> dict[str, float]:
    """Compute OOS classification metrics for one walk-forward step.

    Args:
        preds: Predicted P(up) for the step's bars.
        labels: Realised binary {0,1} outcomes aligned to ``preds``.
        long_threshold: P(up) at/above which the system goes long.
        short_threshold: P(up) at/below which the system goes short.

    Returns:
        Dict with ``oos_accuracy`` (0.5 cut, kept for learning-curve continuity),
        ``oos_logloss``, ``oos_brier``, ``oos_auc`` (NaN if a class is missing),
        ``oos_directional_accuracy`` (accuracy on bars that triggered a long/short
        signal) and ``oos_coverage`` (fraction of bars that triggered a signal).
    """
    p = np.clip(np.asarray(preds, dtype=float), _EPS, 1.0 - _EPS)
    y = np.asarray(labels, dtype=float)
    n = len(p)
    if n == 0:
        return {
            "oos_accuracy": 0.0,
            "oos_logloss": 0.0,
            "oos_brier": 0.0,
            "oos_auc": float("nan"),
            "oos_directional_accuracy": float("nan"),
            "oos_coverage": 0.0,
        }

    accuracy = float(np.mean((p >= 0.5) == (y >= 0.5)))
    logloss = float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))
    brier = float(np.mean((p - y) ** 2))
    auc = _safe_auc(y, p)

    # Threshold-aware: only count bars where the model expresses a directional view.
    long_mask = p >= long_threshold
    short_mask = p <= short_threshold
    decided = long_mask | short_mask
    n_decided = int(decided.sum())
    if n_decided > 0:
        correct = np.sum((long_mask & (y >= 0.5)) | (short_mask & (y < 0.5)))
        directional_accuracy = float(correct / n_decided)
    else:
        directional_accuracy = float("nan")

    return {
        "oos_accuracy": accuracy,
        "oos_logloss": logloss,
        "oos_brier": brier,
        "oos_auc": auc,
        "oos_directional_accuracy": directional_accuracy,
        "oos_coverage": float(n_decided / n),
    }


def regression_step_metrics(preds: np.ndarray, returns: np.ndarray) -> dict[str, float]:
    """Compute OOS regression metrics (directional accuracy + RMSE) for one step."""
    pred = np.asarray(preds, dtype=float)
    ret = np.asarray(returns, dtype=float)
    if len(pred) == 0:
        return {"oos_accuracy": 0.0, "oos_rmse": 0.0}
    # Directional accuracy: did the sign of the predicted return match reality?
    directional = float(np.mean((pred > 0.0) == (ret > 0.0)))
    rmse = float(np.sqrt(np.mean((pred - ret) ** 2)))
    return {"oos_accuracy": directional, "oos_rmse": rmse}


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    """ROC-AUC, or NaN when only one class is present in ``y``."""
    if len(np.unique(y)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score  # noqa: PLC0415 - lazy (core dep)

    return float(roc_auc_score(y, p))

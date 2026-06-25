"""Learning-curve analysis for progressive walk-forward step history."""

from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_step_history(step_history: pd.DataFrame, rolling_window: int = 5) -> dict[str, float]:
    """Summarize OOS learning dynamics from per-step walk-forward metrics.

    Args:
        step_history: DataFrame with at least ``oos_accuracy`` and optionally
            ``oos_logloss`` (as produced by :class:`ProgressiveLearningEngine`).
        rolling_window: Window for rolling mean accuracy/logloss.

    Returns:
        Flat dict suitable for JSON artifacts and MLflow logging.
    """
    if step_history.empty:
        return {"n_steps": 0.0}

    summary: dict[str, float] = {"n_steps": float(len(step_history))}

    if "oos_accuracy" in step_history.columns:
        acc = step_history["oos_accuracy"].astype(float)
        summary.update(
            {
                "mean_oos_accuracy": float(acc.mean()),
                "final_oos_accuracy": float(acc.iloc[-1]),
                "min_oos_accuracy": float(acc.min()),
                "max_oos_accuracy": float(acc.max()),
            }
        )
        if len(acc) >= rolling_window:
            summary["rolling_mean_oos_accuracy"] = float(
                acc.rolling(rolling_window, min_periods=1).mean().iloc[-1]
            )
        if len(acc) >= 2:
            x = np.arange(len(acc), dtype=float)
            slope = float(np.polyfit(x, acc.to_numpy(), 1)[0])
            summary["oos_accuracy_trend_slope"] = slope

    if "oos_logloss" in step_history.columns:
        ll = step_history["oos_logloss"].astype(float)
        summary.update(
            {
                "mean_oos_logloss": float(ll.mean()),
                "final_oos_logloss": float(ll.iloc[-1]),
            }
        )
        if len(ll) >= rolling_window:
            summary["rolling_mean_oos_logloss"] = float(
                ll.rolling(rolling_window, min_periods=1).mean().iloc[-1]
            )

    return summary

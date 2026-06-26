"""Walk-forward learning degradation diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _half_means(step_history: pd.DataFrame, column: str) -> tuple[float, float, float]:
    """Return first-half mean, second-half mean, and second-minus-first delta."""
    half = max(1, len(step_history) // 2)
    first = step_history[column].astype(float).iloc[:half].mean()
    second = step_history[column].astype(float).iloc[half:].mean()
    return float(first), float(second), float(second - first)


def summarize_learning_degradation(step_history: pd.DataFrame) -> dict[str, float]:
    """Summarize first-vs-second-half OOS drift and structural walk-forward signals.

    Expects per-step columns produced by :class:`ProgressiveLearningEngine` including
    ``oos_accuracy``, and optionally ``oos_logloss``, ``oos_directional_accuracy``,
    ``oos_coverage``, ``test_label_rate``, and ``mean_prediction``.
    """
    if step_history.empty or "oos_accuracy" not in step_history.columns:
        return {}

    summary: dict[str, float] = {}

    metric_map = {
        "accuracy": "oos_accuracy",
        "logloss": "oos_logloss",
        "dir_accuracy": "oos_directional_accuracy",
        "coverage": "oos_coverage",
        "label_rate": "test_label_rate",
        "mean_prediction": "mean_prediction",
    }
    for prefix, column in metric_map.items():
        if column not in step_history.columns:
            continue
        first, second, delta = _half_means(step_history, column)
        summary[f"first_half_{prefix}"] = first
        summary[f"second_half_{prefix}"] = second
        summary[f"{prefix}_delta"] = delta

    # Backward-compatible alias used by existing reports/tests.
    if "accuracy_delta" in summary:
        summary["delta"] = summary["accuracy_delta"]

    if "train_rows" in step_history.columns and len(step_history) >= 2:
        train_rows = step_history["train_rows"].astype(float)
        x = np.arange(len(train_rows), dtype=float)
        summary["train_rows_per_step"] = float(np.polyfit(x, train_rows.to_numpy(), 1)[0])
        if "oos_accuracy" in step_history.columns:
            acc = step_history["oos_accuracy"].astype(float)
            if len(acc) >= 2 and acc.std() > 0 and train_rows.std() > 0:
                corr = float(np.corrcoef(train_rows, acc)[0, 1])
                if not np.isnan(corr):
                    summary["accuracy_train_rows_corr"] = corr

    return summary


def degradation_hints(degradation: dict[str, float]) -> list[str]:
    """Turn degradation metrics into short, actionable interpretation lines."""
    if not degradation:
        return []

    hints: list[str] = []

    label_delta = degradation.get("label_rate_delta")
    if label_delta is not None and abs(label_delta) >= 0.03:
        hints.append(
            "OOS up-label rate shifted "
            f"{degradation.get('first_half_label_rate', 0):.2f} → "
            f"{degradation.get('second_half_label_rate', 0):.2f} "
            "(class balance / regime change in the test windows)."
        )

    pred_delta = degradation.get("mean_prediction_delta")
    if pred_delta is not None and abs(pred_delta) >= 0.03:
        hints.append(
            "Mean P(up) drifted "
            f"{degradation.get('first_half_mean_prediction', 0):.2f} → "
            f"{degradation.get('second_half_mean_prediction', 0):.2f} "
            "(calibration may be stale on recent bars)."
        )

    logloss_delta = degradation.get("logloss_delta")
    if logloss_delta is not None and logloss_delta >= 0.05:
        hints.append(
            f"OOS logloss worsened by {logloss_delta:+.3f} "
            "(probabilistic fit degrading, not just threshold accuracy)."
        )

    coverage_delta = degradation.get("coverage_delta")
    if coverage_delta is not None and abs(coverage_delta) >= 0.08:
        hints.append(
            "Threshold-trigger rate changed "
            f"{degradation.get('first_half_coverage', 0):.0%} → "
            f"{degradation.get('second_half_coverage', 0):.0%} "
            "(model confidence / signal frequency shifted)."
        )

    corr = degradation.get("accuracy_train_rows_corr")
    if corr is not None and corr <= -0.25:
        hints.append(
            f"Accuracy falls as train_rows grow (corr={corr:+.2f}) — "
            "expanding history may dilute recent regimes; try lowering "
            "recency_half_life or capping train window."
        )

    acc_delta = degradation.get("accuracy_delta", degradation.get("delta", 0.0))
    if acc_delta <= -0.05 and not hints:
        hints.append(
            "OOS accuracy declined without a clear label or calibration shift — "
            "early walk-forward steps may have overfit a regime that later reversed."
        )

    return hints

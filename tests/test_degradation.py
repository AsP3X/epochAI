"""Tests for walk-forward learning degradation diagnostics."""

from __future__ import annotations

import pandas as pd
import pytest

from epoch_ai.learning.degradation import degradation_hints, summarize_learning_degradation


def test_summarize_learning_degradation_half_split():
    step_history = pd.DataFrame(
        {
            "oos_accuracy": [0.60, 0.58, 0.56, 0.50, 0.48, 0.46],
            "oos_logloss": [0.80, 0.82, 0.85, 0.92, 0.95, 0.98],
            "oos_directional_accuracy": [0.62, 0.60, 0.58, 0.52, 0.50, 0.48],
            "oos_coverage": [0.55, 0.56, 0.57, 0.62, 0.63, 0.64],
            "test_label_rate": [0.52, 0.51, 0.50, 0.46, 0.45, 0.44],
            "mean_prediction": [0.50, 0.51, 0.51, 0.54, 0.55, 0.56],
            "train_rows": [1800, 2000, 2200, 2400, 2600, 2800],
        }
    )
    summary = summarize_learning_degradation(step_history)

    assert summary["first_half_accuracy"] > summary["second_half_accuracy"]
    assert summary["delta"] == summary["accuracy_delta"]
    assert summary["logloss_delta"] > 0
    assert summary["train_rows_per_step"] == pytest.approx(200.0)
    assert summary["accuracy_train_rows_corr"] < 0


def test_degradation_hints_cover_regime_and_logloss():
    summary = summarize_learning_degradation(
        pd.DataFrame(
            {
                "oos_accuracy": [0.56] * 3 + [0.49] * 3,
                "oos_logloss": [0.85] * 3 + [0.95] * 3,
                "test_label_rate": [0.55] * 3 + [0.45] * 3,
                "mean_prediction": [0.50] * 3 + [0.56] * 3,
                "train_rows": [2000, 2200, 2400, 2600, 2800, 3000],
            }
        )
    )
    hints = degradation_hints(summary)
    joined = " ".join(hints)
    assert "up-label rate" in joined
    assert "logloss" in joined.lower()

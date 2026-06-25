"""Tests for learning-curve analysis."""

from __future__ import annotations

import pandas as pd

from epoch_ai.learning.curve_analysis import summarize_step_history


def test_summarize_step_history_empty():
    assert summarize_step_history(pd.DataFrame()) == {"n_steps": 0.0}


def test_summarize_step_history_trend():
    df = pd.DataFrame(
        {
            "oos_accuracy": [0.4, 0.45, 0.5, 0.55, 0.6],
            "oos_logloss": [0.7, 0.68, 0.65, 0.63, 0.6],
        }
    )
    summary = summarize_step_history(df, rolling_window=3)
    assert summary["n_steps"] == 5.0
    assert summary["mean_oos_accuracy"] == 0.5
    assert summary["final_oos_accuracy"] == 0.6
    assert summary["oos_accuracy_trend_slope"] > 0

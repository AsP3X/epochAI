"""Tests for backtest report formatting helpers."""

from __future__ import annotations

import pandas as pd

from epoch_ai.backtesting.reporting import (
    count_rebalances,
    format_importance_value,
    importance_metric_label,
)


def test_importance_metric_label_by_backend():
    assert importance_metric_label("evolved_nn") == "permutation importance"
    assert importance_metric_label("lightgbm") == "gain"


def test_format_importance_value_preserves_small_scores():
    assert format_importance_value(0.0) == "0"
    assert format_importance_value(123.4) == "123.4"
    assert format_importance_value(0.0456) == "0.0456"
    assert "e" in format_importance_value(0.00012)


def test_count_rebalances_counts_weight_changes():
    weights = pd.Series([0.0, 0.5, 0.5, -0.5, -0.5, 0.0])
    assert count_rebalances(weights, horizon_aware=False, horizon=1) == 3
    assert count_rebalances(weights, horizon_aware=True, horizon=2) >= 1

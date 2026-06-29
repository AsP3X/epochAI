"""Automated backtest smoke tests."""

from __future__ import annotations

from epoch_ai.backtesting.engine import Backtester
from epoch_ai.features.pipeline import FeaturePipeline

import pytest

pytestmark = pytest.mark.slow


def test_backtest_learning_curve(market, small_config):
    features = FeaturePipeline(small_config).transform(market)
    result = Backtester(small_config).run(market, features=features)

    assert result.learning_curve.get("n_steps", 0) >= 1
    assert "mean_oos_accuracy" in result.learning_curve
    if result.learning_improvement:
        assert "first_half_accuracy" in result.learning_improvement
        assert "delta" in result.learning_improvement
    assert "n_rebalances" in result.metrics
    assert result.metrics["n_rebalances"] <= len(result.learning.predictions)

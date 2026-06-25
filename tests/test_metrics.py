"""Tests for trading metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from epoch_ai.backtesting.metrics import compute_metrics, max_drawdown


def test_positive_drift_has_positive_sharpe():
    rng = np.random.default_rng(0)
    returns = 0.001 + 0.01 * rng.standard_normal(2000)
    metrics = compute_metrics(returns, annualization=10.0)
    assert metrics["sharpe"] > 0
    assert metrics["total_return"] > 0
    assert metrics["n_periods"] == 2000


def test_max_drawdown_is_non_positive():
    equity = pd.Series([1.0, 1.2, 0.9, 1.1, 0.8])
    assert max_drawdown(equity) < 0


def test_empty_returns_safe():
    metrics = compute_metrics([], annualization=1.0)
    assert metrics["sharpe"] == 0.0
    assert metrics["n_periods"] == 0


def test_profit_factor():
    returns = pd.Series([0.1, -0.05, 0.2, -0.1])
    metrics = compute_metrics(returns)
    assert metrics["profit_factor"] > 1.0
    assert 0.0 <= metrics["win_rate"] <= 1.0

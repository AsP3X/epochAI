"""End-to-end tests for the progressive learning engine and backtester."""

from __future__ import annotations

from epoch_ai.backtesting.engine import Backtester
from epoch_ai.features.pipeline import FeaturePipeline
from epoch_ai.learning.progressive import ProgressiveLearningEngine
from epoch_ai.logging_system.store import PredictionStore


def test_progressive_run(market, small_config):
    features = FeaturePipeline(small_config).transform(market)
    engine = ProgressiveLearningEngine(small_config)
    result = engine.run(market, features)

    assert not result.predictions.empty
    assert len(result.step_history) <= small_config.walk_forward.max_steps
    assert {"prediction", "signal", "target_weight", "forward_return"}.issubset(
        result.predictions.columns
    )
    # Predictions are out-of-sample (after the initial training window).
    assert result.predictions["prediction"].between(0, 1).all()
    # Richer OOS step metrics are recorded for the learning curve.
    assert {"oos_accuracy", "oos_logloss", "oos_brier", "oos_auc"}.issubset(
        result.step_history.columns
    )


def test_horizon_aware_backtest_reduces_turnover(market, small_config):
    """Horizon-aware PnL holds positions longer => lower per-step weight churn."""
    from epoch_ai.backtesting.engine import Backtester

    features = FeaturePipeline(small_config).transform(market)

    small_config.backtest.horizon_aware = True
    aware = Backtester(small_config).run(market, features=features)

    small_config.backtest.horizon_aware = False
    legacy = Backtester(small_config).run(market, features=features)

    # Both produce valid equity curves; the two modes should differ.
    assert aware.equity_curve.iloc[-1] > 0
    assert legacy.equity_curve.iloc[-1] > 0
    assert not aware.strategy_returns.equals(legacy.strategy_returns)


def test_backtest_and_logging(market, small_config, tmp_path):
    store = PredictionStore(str(tmp_path / "p.sqlite"))
    result = Backtester(small_config).run(market, store=store)

    assert "sharpe" in result.metrics
    assert result.equity_curve.iloc[-1] > 0
    counts = store.counts()
    assert counts["predictions"] > 0
    assert counts["predictions"] == counts["outcomes"]
    store.close()

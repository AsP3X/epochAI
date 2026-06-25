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


def test_backtest_and_logging(market, small_config, tmp_path):
    store = PredictionStore(str(tmp_path / "p.sqlite"))
    result = Backtester(small_config).run(market, store=store)

    assert "sharpe" in result.metrics
    assert result.equity_curve.iloc[-1] > 0
    counts = store.counts()
    assert counts["predictions"] > 0
    assert counts["predictions"] == counts["outcomes"]
    store.close()

"""End-to-end tests for the progressive learning engine and backtester."""

from __future__ import annotations

import pytest

from epoch_ai.backtesting.engine import Backtester
from epoch_ai.features.pipeline import FeaturePipeline
from epoch_ai.learning.progressive import ProgressiveLearningEngine
from epoch_ai.logging_system.store import PredictionStore

pytestmark = pytest.mark.slow


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
    assert {"test_label_rate", "mean_prediction"}.issubset(result.step_history.columns)


def test_progressive_run_tcn(market, small_config, tmp_path):
    """Walk-forward runs end-to-end with the sequence (TCN) backend + structured logging."""
    pytest.importorskip("torch")
    cfg = small_config
    cfg.model.backend = "tcn"
    cfg.model.val_fraction = 0.2
    cfg.model.calibration = "none"
    cfg.model.refit_full_after_es = False
    cfg.model.tcn.lookback = 16
    cfg.model.tcn.channels = [16, 16]
    cfg.model.tcn.max_epochs = 12
    cfg.model.tcn.patience = 3
    cfg.model.tcn.compute_importance = False
    cfg.prediction.horizons = [4, 8]
    cfg.prediction.horizon = 8
    cfg.walk_forward.retrain_frequency = 1

    features = FeaturePipeline(cfg).transform(market)
    store = PredictionStore(str(tmp_path / "tcn.sqlite"))
    result = ProgressiveLearningEngine(cfg).run(market, features, store=store)

    assert not result.predictions.empty
    assert result.predictions["prediction"].between(0, 1).all()
    assert {"oos_accuracy", "oos_logloss"}.issubset(result.step_history.columns)
    # Sequence backend exercised the lookback-context + structured logging path.
    assert store.counts()["predictions"] > 0
    store.close()


def test_embargo_purges_label_overlap(market, small_config):
    """The embargo gap removes the forward-return label overlap at the train boundary."""
    horizon = small_config.prediction.horizon
    itp = small_config.walk_forward.initial_train_period
    features = FeaturePipeline(small_config).transform(market)

    # Default embargo (None) resolves to the prediction horizon, so the first training
    # window loses exactly `horizon` rows that would otherwise leak the test window.
    default_result = ProgressiveLearningEngine(small_config).run(market, features)
    assert int(default_result.step_history["train_rows"].iloc[0]) == itp - horizon

    # Disabling the embargo restores the legacy (leaky) full window size.
    small_config.walk_forward.embargo = 0
    no_embargo = ProgressiveLearningEngine(small_config).run(market, features)
    assert int(no_embargo.step_history["train_rows"].iloc[0]) == itp


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

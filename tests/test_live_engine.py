"""Tests for live trading engine (simulated feed)."""

from __future__ import annotations

import pytest

from epoch_ai.services.runtime import RuntimeService
from epoch_ai.services.training import TrainingService

pytestmark = pytest.mark.slow


def test_live_feed_predict_and_trade(small_config, tmp_path):
    small_config.model.model_dir = str(tmp_path / "models")
    small_config.data.data_dir = str(tmp_path / "data")
    small_config.logging.db_path = str(tmp_path / "pred.sqlite")
    small_config.execution.min_buffer_bars = 800
    small_config.walk_forward.initial_train_period = 800
    small_config.execution.reserve_fraction = 0.2

    TrainingService(small_config).train(n_bars=2500, max_steps=2, register=True)

    runtime = RuntimeService(small_config)
    result = runtime.run_live_feed(
        n_bars=2500,
        feed_bars=50,
        log_predictions=True,
    )
    assert result.ticks > 0
    assert result.model_version.startswith("v_")

"""Golden benchmark: reproducible train + live-feed smoke with metric bounds."""

from __future__ import annotations

import pytest

from epoch_ai.services.runtime import RuntimeService
from epoch_ai.services.training import TrainingService


@pytest.mark.slow
def test_golden_train_and_live_feed(small_config, tmp_path):
    """End-to-end benchmark with fixed synthetic seed and loose stability checks."""
    small_config.model.model_dir = str(tmp_path / "models")
    small_config.data.data_dir = str(tmp_path / "data")
    small_config.data.synthetic_seed = 42
    small_config.logging.db_path = str(tmp_path / "pred.sqlite")
    small_config.execution.min_buffer_bars = 800
    small_config.walk_forward.initial_train_period = 800

    train = TrainingService(small_config).train(n_bars=2500, max_steps=2, register=True)
    assert train.model_version is not None
    assert train.walk_forward_steps >= 1

    runtime = RuntimeService(small_config)
    result = runtime.run_live_feed(n_bars=2500, feed_bars=30, log_predictions=True)
    assert result.ticks >= 10
    assert result.final_equity > 0
    assert result.model_version == train.model_version

"""Tests for live engine hardening."""

from __future__ import annotations

import pytest

from epoch_ai.execution.kill_switch import KillSwitch
from epoch_ai.execution.live_engine import LiveTradingEngine
from epoch_ai.services.training import TrainingService

pytestmark = pytest.mark.slow


def test_live_engine_respects_kill_switch(small_config, tmp_path):
    small_config.model.model_dir = str(tmp_path / "models")
    small_config.data.data_dir = str(tmp_path / "data")
    small_config.execution.min_buffer_bars = 800
    small_config.walk_forward.initial_train_period = 800
    small_config.execution.kill_switch_path = str(tmp_path / "kill.json")

    TrainingService(small_config).train(n_bars=2500, max_steps=2, register=True)
    KillSwitch(small_config.execution.kill_switch_path).halt("test")

    engine = LiveTradingEngine.create(small_config)
    from epoch_ai.data.downloader import HistoricalDownloader

    market = HistoricalDownloader(small_config).load_or_download(n_bars=2500)
    tick = engine.process_bar(small_config.primary_symbol, market.iloc[:900])
    assert tick is not None
    assert tick.halted
    assert tick.fill is None

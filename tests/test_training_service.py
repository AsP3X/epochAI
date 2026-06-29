"""Tests for TrainingService real-data policy."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import pandas as pd

from epoch_ai.config.settings import AppConfig
from epoch_ai.services.training import TrainingService, minimum_training_bars, resolve_training_bars


def test_evolved_nn_disables_synthetic_fallback_in_download_config():
    cfg = AppConfig.model_validate(
        {
            "data": {"use_synthetic_fallback": True},
            "model": {"backend": "evolved_nn"},
        }
    )
    resolved = TrainingService(cfg)._training_data_config()
    assert resolved.data.use_synthetic_fallback is False


def test_lightgbm_keeps_synthetic_fallback_setting():
    cfg = AppConfig.model_validate(
        {
            "data": {"use_synthetic_fallback": True},
            "model": {"backend": "lightgbm"},
        }
    )
    resolved = TrainingService(cfg)._training_data_config()
    assert resolved.data.use_synthetic_fallback is True


def test_minimum_training_bars_matches_config():
    cfg = AppConfig.model_validate(
        {
            "walk_forward": {"initial_train_period": 43200, "embargo": None},
            "execution": {"min_buffer_bars": 7500},
            "prediction": {"horizons": [1, 5, 10, 15, 30, 60]},
        }
    )
    assert minimum_training_bars(cfg) == int((43200 + 1) / 0.50) + 60 + 500


def test_train_rejects_bars_below_minimum():
    cfg = AppConfig.model_validate(
        {
            "walk_forward": {"initial_train_period": 800, "embargo": 0},
            "execution": {"min_buffer_bars": 500},
            "prediction": {"horizon": 10, "horizons": [1, 5, 10]},
        }
    )
    service = TrainingService(cfg)
    with pytest.raises(ValueError, match="too small for training"):
        service.train(n_bars=500, max_steps=1, register=False)


def test_resolve_training_bars_uses_live_cache(tmp_path):
    cfg = AppConfig.model_validate(
        {
            "data": {"data_dir": str(tmp_path / "data")},
            "walk_forward": {"initial_train_period": 800, "embargo": 0},
            "execution": {"min_buffer_bars": 500},
            "prediction": {"horizon": 10, "horizons": [1, 5, 10]},
        }
    )
    from epoch_ai.data.downloader import HistoricalDownloader

    end = pd.Timestamp.now(tz="UTC").floor("1min")
    index = pd.date_range(end=end, periods=52000, freq="1min", tz="UTC")
    close = pd.Series(100.0, index=index)
    frame = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 1.0},
        index=index,
    )
    downloader = HistoricalDownloader(cfg)
    cache_path = downloader._cache_path(cfg.primary_symbol)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache_path)

    assert resolve_training_bars(cfg, None) == 52000
    assert resolve_training_bars(cfg, None, full_history=True) is None
    assert resolve_training_bars(cfg, 8000) == 8000


def test_train_uses_cache_only_by_default():
    cfg = AppConfig.model_validate(
        {
            "walk_forward": {"initial_train_period": 800, "embargo": 0},
            "execution": {"min_buffer_bars": 500},
            "prediction": {"horizon": 10, "horizons": [1, 5, 10]},
        }
    )
    service = TrainingService(cfg)
    with patch.object(service, "download") as mock_download:
        mock_download.side_effect = RuntimeError("cache too small")
        with pytest.raises(RuntimeError, match="cache too small"):
            service.train(n_bars=87000, max_steps=1, register=False)
        mock_download.assert_called_once_with(
            n_bars=87000,
            force=False,
            fetch_if_missing=False,
        )


def test_train_refresh_data_fetches_from_exchange():
    cfg = AppConfig.model_validate(
        {
            "walk_forward": {"initial_train_period": 800, "embargo": 0},
            "execution": {"min_buffer_bars": 500},
            "prediction": {"horizon": 10, "horizons": [1, 5, 10]},
        }
    )
    service = TrainingService(cfg)
    with patch.object(service, "download") as mock_download:
        mock_download.side_effect = RuntimeError("stop after download")
        with pytest.raises(RuntimeError, match="stop after download"):
            service.train(
                n_bars=87000,
                max_steps=1,
                register=False,
                refresh_data=True,
            )
        mock_download.assert_called_once_with(
            n_bars=87000,
            force=True,
            fetch_if_missing=True,
        )

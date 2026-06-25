"""Tests for configuration loading and validation."""

from __future__ import annotations

import pytest
import yaml

from epoch_ai.config.settings import AppConfig, load_config


def test_defaults_are_valid():
    config = AppConfig()
    assert config.primary_symbol == "BTC/USDT"
    assert config.prediction.horizon >= 1


def test_load_from_yaml(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"symbols": ["ETH/USDT"], "timeframe": "5m"}))
    config = load_config(path)
    assert config.primary_symbol == "ETH/USDT"
    assert config.timeframe == "5m"


def test_invalid_horizon_rejected():
    with pytest.raises(ValueError):
        AppConfig.model_validate({"prediction": {"horizon": 0}})


def test_initial_train_must_exceed_horizon():
    with pytest.raises(ValueError):
        AppConfig.model_validate(
            {"prediction": {"horizon": 50}, "walk_forward": {"initial_train_period": 10}}
        )


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_new_model_and_feature_defaults():
    config = AppConfig()
    assert config.model.class_weight == "balanced"
    assert config.model.calibration == "isotonic"
    assert 0.0 <= config.model.val_fraction < 0.5
    assert config.backtest.horizon_aware is True
    assert config.features.ma_windows
    assert config.features.rsi_periods


def test_empty_feature_window_rejected():
    with pytest.raises(ValueError):
        AppConfig.model_validate({"features": {"ma_windows": []}})


def test_invalid_val_fraction_rejected():
    with pytest.raises(ValueError):
        AppConfig.model_validate({"model": {"val_fraction": 0.9}})


def test_shipped_config_yaml_loads():
    """The example config must resolve with the new keys."""
    config = load_config("config/config.yaml")
    assert config.model.calibration in {"none", "isotonic", "sigmoid"}
    assert config.walk_forward.recency_half_life == 4000

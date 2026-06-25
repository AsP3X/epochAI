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

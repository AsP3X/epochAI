"""Tests for config overrides."""

from __future__ import annotations

import pytest

from epoch_ai.config.overrides import apply_overrides, parse_set_args
from epoch_ai.config.settings import AppConfig


def test_apply_overrides_nested():
    base = {"walk_forward": {"step_size": 200}, "timeframe": "15m"}
    merged = apply_overrides(base, {"walk_forward.step_size": 100})
    assert merged["walk_forward"]["step_size"] == 100
    assert merged["timeframe"] == "15m"


def test_parse_set_args():
    overrides = parse_set_args(["timeframe=5m", "walk_forward.max_steps=3"])
    config = AppConfig.model_validate(apply_overrides({}, overrides))
    assert config.timeframe == "5m"
    assert config.walk_forward.max_steps == 3


def test_parse_set_args_invalid():
    with pytest.raises(ValueError):
        parse_set_args(["badtoken"])

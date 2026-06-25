"""Shared pytest fixtures."""

from __future__ import annotations

import pandas as pd
import pytest

from epoch_ai.config.settings import AppConfig
from epoch_ai.data.synthetic import generate_synthetic_ohlcv


@pytest.fixture(scope="session")
def market() -> pd.DataFrame:
    """A reproducible synthetic market with derivatives context."""
    return generate_synthetic_ohlcv(timeframe="15m", start="2020-01-01", n_bars=4000, seed=11)


@pytest.fixture
def small_config() -> AppConfig:
    """A fast config suitable for unit tests."""
    return AppConfig.model_validate(
        {
            "symbols": ["BTC/USDT"],
            "timeframe": "15m",
            "prediction": {"horizon": 8},
            "model": {"num_boost_round": 40, "early_stopping_rounds": None},
            "walk_forward": {
                "initial_train_period": 800,
                "step_size": 400,
                "retrain_frequency": 1,
                "max_steps": 3,
            },
        }
    )

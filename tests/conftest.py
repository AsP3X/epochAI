"""Shared pytest fixtures."""

from __future__ import annotations

import pandas as pd
import pytest

from epoch_ai.config.settings import AppConfig
from epoch_ai.data.downloader import HistoricalDownloader
from epoch_ai.data.synthetic import generate_synthetic_ohlcv

# Modules that must hit the real downloader (CCXT/synthetic/cache behaviour).
_DOWNLOADER_INTEGRATION_MODULES = frozenset(
    {
        "tests.test_downloader",
        "tests.test_enrichment",
        "tests.test_cli",
    }
)


@pytest.fixture(scope="session")
def market() -> pd.DataFrame:
    """A reproducible synthetic market with derivatives context."""
    return generate_synthetic_ohlcv(timeframe="15m", start="2020-01-01", n_bars=4000, seed=11)


@pytest.fixture(autouse=True)
def _isolate_walk_forward_checkpoints(tmp_path, monkeypatch):
    """Redirect auto-named walk-forward checkpoints to a per-test temp directory.

    Tests that don't set ``walk_forward.checkpoint_path`` otherwise resolve to the real
    ``artifacts/checkpoints/`` tree and leak state across tests (and into the user's
    workspace). Pointing the default dir at ``tmp_path`` keeps every test isolated.
    """
    import epoch_ai.learning.checkpoint as checkpoint_mod

    monkeypatch.setattr(
        checkpoint_mod,
        "DEFAULT_CHECKPOINT_DIR",
        tmp_path / "checkpoints",
        raising=True,
    )


@pytest.fixture(autouse=True)
def _patch_offline_market_download(request, monkeypatch, market):
    """Keep unit/integration tests offline-fast; real CCXT is tested in downloader/cli."""
    module = request.node.module.__name__
    if module in _DOWNLOADER_INTEGRATION_MODULES:
        return

    def fake_load(self, symbol=None, *, n_bars=None, force=False, skip_enrichment=False):
        cap = len(market) if n_bars is None else min(n_bars, len(market))
        return market.iloc[:cap].copy()

    monkeypatch.setattr(HistoricalDownloader, "load_or_download", fake_load)


@pytest.fixture
def small_config() -> AppConfig:
    """A fast config suitable for unit tests."""
    return AppConfig.model_validate(
        {
            "symbols": ["BTC/USDT"],
            "timeframe": "15m",
            "prediction": {"horizon": 8},
            "data": {"synthesize_market_extensions": False},
            "features": {
                "higher_timeframe": False,
                "macro": False,
                "onchain": False,
                "patterns": False,
                "manipulation": False,
            },
            "model": {
                "backend": "lightgbm",
                "num_boost_round": 40,
                "early_stopping_rounds": None,
            },
            "walk_forward": {
                "initial_train_period": 800,
                "step_size": 400,
                "retrain_frequency": 1,
                "max_steps": 3,
            },
        }
    )


@pytest.fixture
def pattern_config(small_config: AppConfig) -> AppConfig:
    """Config with pattern and manipulation groups enabled."""
    small_config.features.patterns = True
    small_config.features.manipulation = True
    return small_config

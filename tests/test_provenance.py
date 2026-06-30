"""Tests for parquet cache provenance and real-data training policy."""

from __future__ import annotations

import pytest

from epoch_ai.config.settings import AppConfig
from epoch_ai.data.provenance import (
    SOURCE_EXCHANGE,
    SOURCE_SYNTHETIC,
    assert_cache_is_real,
    provenance_path,
    write_data_provenance,
)
from epoch_ai.data.training_policy import config_for_supervised_training


def test_config_for_supervised_training_disables_synthetic_all_backends():
    for backend in ("evolved_nn", "lightgbm", "xgboost"):
        cfg = AppConfig.model_validate(
            {
                "data": {"use_synthetic_fallback": True},
                "model": {"backend": backend},
            }
        )
        resolved = config_for_supervised_training(cfg)
        assert resolved.data.use_synthetic_fallback is False


def test_assert_cache_is_real_rejects_synthetic(tmp_path):
    cache = tmp_path / "BTC-USDT_1m.parquet"
    cache.write_bytes(b"stub")
    write_data_provenance(
        cache,
        source=SOURCE_SYNTHETIC,
        symbol="BTC/USDT",
        timeframe="1m",
        n_bars=100,
    )
    with pytest.raises(RuntimeError, match="synthetic"):
        assert_cache_is_real(cache, symbol="BTC/USDT")


def test_assert_cache_is_real_accepts_exchange(tmp_path):
    cache = tmp_path / "BTC-USDT_1m.parquet"
    cache.write_bytes(b"stub")
    write_data_provenance(
        cache,
        source=SOURCE_EXCHANGE,
        symbol="BTC/USDT",
        timeframe="1m",
        n_bars=100,
    )
    assert_cache_is_real(cache, symbol="BTC/USDT")


def test_assert_cache_is_real_rejects_missing_provenance(tmp_path):
    cache = tmp_path / "BTC-USDT_1m.parquet"
    cache.write_bytes(b"stub")
    with pytest.raises(RuntimeError, match="no provenance"):
        assert_cache_is_real(cache, symbol="BTC/USDT")


def test_provenance_sidecar_path():
    assert provenance_path(
        __import__("pathlib").Path("artifacts/data/BTC-USDT_1m.parquet")
    ).name == "BTC-USDT_1m.provenance.json"

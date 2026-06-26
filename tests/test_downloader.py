"""Tests for historical downloader fallback and cache behaviour."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from epoch_ai.config.settings import AppConfig
from epoch_ai.data.downloader import HistoricalDownloader


def _ohlcv_frame(n_bars: int, *, start: str = "2019-11-01") -> pd.DataFrame:
    index = pd.date_range(start=start, periods=n_bars, freq="15min", tz="UTC")
    close = pd.Series(range(n_bars), dtype=float, index=index) + 100.0
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1.0,
        },
        index=index,
    )


def test_synthetic_fallback_when_ccxt_unavailable(tmp_path, monkeypatch):
    config = AppConfig.model_validate(
        {
            "data": {
                "data_dir": str(tmp_path / "data"),
                "use_synthetic_fallback": True,
            }
        }
    )
    monkeypatch.setattr(
        HistoricalDownloader,
        "_download_ccxt",
        lambda *args, **kwargs: None,
    )
    df = HistoricalDownloader(config).load_or_download(n_bars=500)
    assert len(df) == 500
    assert (tmp_path / "data" / "BTC-USDT_15m.parquet").exists()


def test_synthetic_fallback_disabled_raises_without_cache(tmp_path, monkeypatch):
    config = AppConfig.model_validate(
        {
            "data": {
                "data_dir": str(tmp_path / "data"),
                "use_synthetic_fallback": False,
            }
        }
    )
    monkeypatch.setattr(
        HistoricalDownloader,
        "_download_ccxt",
        lambda *args, **kwargs: None,
    )
    with pytest.raises(RuntimeError, match="synthetic fallback disabled"):
        HistoricalDownloader(config).load_or_download(n_bars=500)


def test_uses_cached_real_data_when_extension_fails(tmp_path, monkeypatch):
    config = AppConfig.model_validate(
        {
            "data": {
                "data_dir": str(tmp_path / "data"),
                "use_synthetic_fallback": False,
            }
        }
    )
    downloader = HistoricalDownloader(config)
    cached = _ohlcv_frame(200)
    cache_path = downloader._cache_path(config.primary_symbol)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached.to_parquet(cache_path)

    monkeypatch.setattr(
        HistoricalDownloader,
        "_download_ccxt",
        lambda *args, **kwargs: None,
    )
    df = downloader.load_or_download(n_bars=500)
    assert len(df) == 200


def test_returns_recent_tail_from_cache(tmp_path, monkeypatch):
    config = AppConfig.model_validate({"data": {"data_dir": str(tmp_path / "data")}})
    downloader = HistoricalDownloader(config)
    cached = _ohlcv_frame(1000)
    cache_path = downloader._cache_path(config.primary_symbol)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached.to_parquet(cache_path)

    def fail_download(*args, **kwargs):
        raise AssertionError("should not download when cache is sufficient")

    monkeypatch.setattr(HistoricalDownloader, "_download_ccxt", fail_download)
    df = downloader.load_or_download(n_bars=500)
    # Returns the most recent 500 bars (not the whole cache), without re-downloading.
    assert len(df) == 500
    assert df.index.max() == cached.index.max()
    assert df.index.min() == cached.index[-500]
    # Full history remains cached on disk.
    assert len(pd.read_parquet(cache_path)) == 1000


def test_returns_full_cache_when_n_bars_none(tmp_path, monkeypatch):
    config = AppConfig.model_validate({"data": {"data_dir": str(tmp_path / "data")}})
    downloader = HistoricalDownloader(config)
    cached = _ohlcv_frame(1000)
    cache_path = downloader._cache_path(config.primary_symbol)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached.to_parquet(cache_path)
    monkeypatch.setattr(
        HistoricalDownloader,
        "_download_ccxt",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no download")),
    )
    df = downloader.load_or_download(n_bars=None)
    assert len(df) == 1000


def test_extends_cache_instead_of_redownloading(tmp_path):
    config = AppConfig.model_validate({"data": {"data_dir": str(tmp_path / "data")}})
    downloader = HistoricalDownloader(config)
    cached = _ohlcv_frame(200)
    cache_path = downloader._cache_path(config.primary_symbol)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached.to_parquet(cache_path)

    last_ms = int(cached.index.max().timestamp() * 1000)
    tf_ms = 15 * 60_000

    def fake_download_ccxt(self, symbol, target_bars, *, base_df=None):
        assert base_df is not None
        assert len(base_df) == 200
        since_ms = int(base_df.index.max().timestamp() * 1000) + tf_ms
        rows = [[since_ms + i * tf_ms, 1.0, 2.0, 0.5, 1.5, 10.0] for i in range(100)]
        new_df = HistoricalDownloader._rows_to_dataframe(rows)
        return pd.concat([base_df, new_df]).iloc[:target_bars]

    with patch.object(HistoricalDownloader, "_download_ccxt", fake_download_ccxt):
        df = downloader.load_or_download(n_bars=250)

    assert len(df) == 250
    assert len(pd.read_parquet(cache_path)) == 250
    assert last_ms < int(df.index[200].timestamp() * 1000)


def test_open_interest_since_clamped_to_exchange_window(tmp_path):
    """Binance openInterestHist rejects startTime older than ~30 days."""
    config = AppConfig.model_validate({"data": {"data_dir": str(tmp_path / "data")}})
    downloader = HistoricalDownloader(config)
    df = _ohlcv_frame(4000)  # ~41 days at 15m — wider than Binance OI retention
    end_ms = int(df.index.max().timestamp() * 1000)
    captured_since: list[int] = []

    class FakeExchange:
        has = {"fetchOpenInterestHistory": True}

        def parse8601(self, iso: str) -> int:
            return int(pd.Timestamp(iso).timestamp() * 1000)

        def fetch_open_interest_history(self, symbol, timeframe, since, limit):
            captured_since.append(since)
            return [
                {
                    "timestamp": since + 15 * 60 * 1000,
                    "openInterestValue": 100.0,
                    "openInterestAmount": 100.0,
                }
            ]

    result = downloader._attach_open_interest(FakeExchange(), "BTC/USDT", df.copy())
    assert captured_since, "expected at least one OI fetch"
    assert captured_since[0] >= end_ms - 30 * 24 * 60 * 60 * 1000
    assert captured_since[0] < end_ms
    assert "open_interest" in result.columns

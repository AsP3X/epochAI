"""Tests for historical downloader fallback and cache behaviour."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from epoch_ai.config.settings import AppConfig
from epoch_ai.data.downloader import HistoricalDownloader


def _ohlcv_frame(n_bars: int, *, start: str = "2019-11-01", live: bool = False) -> pd.DataFrame:
    if live:
        end = pd.Timestamp.now(tz="UTC").floor("15min")
        index = pd.date_range(end=end, periods=n_bars, freq="15min", tz="UTC")
    else:
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
    cache = tmp_path / "data" / "BTC-USDT_15m.parquet"
    assert cache.exists()
    from epoch_ai.data.provenance import read_data_provenance

    meta = read_data_provenance(cache)
    assert meta is not None
    assert meta["source"] == "synthetic"


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
    cached = _ohlcv_frame(1000, live=True)
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
    with patch.object(downloader, "_default_bar_count", return_value=1000):
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

    def fake_download_ccxt(self, symbol, target_bars, *, base_df=None, since_ts=None, until_ts=None, recent_tail=False):
        del symbol, since_ts, until_ts, recent_tail
        assert base_df is not None
        since_ms = int(base_df.index.max().timestamp() * 1000) + tf_ms
        rows = [[since_ms + i * tf_ms, 1.0, 2.0, 0.5, 1.5, 10.0] for i in range(100)]
        new_df = HistoricalDownloader._rows_to_dataframe(rows)
        return pd.concat([base_df, new_df]).iloc[:target_bars]

    with patch.object(HistoricalDownloader, "_download_ccxt", fake_download_ccxt):
        with patch.object(downloader, "_default_bar_count", return_value=250):
            df = downloader.load_or_download(n_bars=None)

    assert len(df) == 250
    assert len(pd.read_parquet(cache_path)) == 250
    assert last_ms < int(df.index[200].timestamp() * 1000)


def test_live_cache_at_exchange_earliest_skips_download(tmp_path):
    """A live cache that already reaches the exchange's earliest candle is complete.

    Regression: with an ``earliest`` start that resolves earlier than the exchange's
    first available candle, the cache can never reach the configured start date. The
    downloader used to treat that as "incomplete" and re-extend (re-fetch + re-attach
    funding/OI + re-save) on *every* run. A current cache at the exchange floor must be
    returned as-is without any download.
    """
    config = AppConfig.model_validate(
        {
            "data": {
                "data_dir": str(tmp_path / "data"),
                "historical_start_date": "earliest",
            }
        }
    )
    downloader = HistoricalDownloader(config)
    cached = _ohlcv_frame(2000, live=True)  # begins long after the 2017 fallback start
    cache_path = downloader._cache_path(config.primary_symbol)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached.to_parquet(cache_path)

    with patch.object(
        HistoricalDownloader,
        "_exchange_earliest_ts",
        lambda self, sym: cached.index.min(),
    ):
        with patch.object(
            HistoricalDownloader,
            "_download_ccxt",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not download")),
        ):
            with patch.object(downloader, "_default_bar_count", return_value=10_000_000):
                df = downloader.load_or_download(n_bars=None)

    assert len(df) == 2000  # full cache returned, untouched


def test_extends_forward_when_cache_reaches_exchange_earliest(tmp_path):
    """A *stale* cache at the exchange floor extends forward, never re-backfills.

    When the cache reaches the exchange's earliest candle but has fallen behind the
    present, the downloader must fetch only the missing tail (passing the cached frame
    as a base) rather than re-downloading the entire history from scratch.
    """
    config = AppConfig.model_validate(
        {
            "data": {
                "data_dir": str(tmp_path / "data"),
                "historical_start_date": "earliest",
            }
        }
    )
    downloader = HistoricalDownloader(config)
    cached = _ohlcv_frame(2000, live=False)  # begins after 2017 fallback, ends in the past
    cache_path = downloader._cache_path(config.primary_symbol)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached.to_parquet(cache_path)

    tf_ms = 15 * 60_000

    def fake_download_ccxt(self, symbol, target_bars, *, base_df=None, since_ts=None, until_ts=None, recent_tail=False):
        del symbol, target_bars, since_ts, until_ts, recent_tail
        # Extend-forward path passes the cached frame; a full re-backfill would not.
        assert base_df is not None, "must extend cache, not re-backfill from scratch"
        since_ms = int(base_df.index.max().timestamp() * 1000) + tf_ms
        rows = [[since_ms + i * tf_ms, 1.0, 2.0, 0.5, 1.5, 10.0] for i in range(3)]
        new_df = HistoricalDownloader._rows_to_dataframe(rows)
        return pd.concat([base_df, new_df])

    with patch.object(
        HistoricalDownloader,
        "_exchange_earliest_ts",
        lambda self, sym: cached.index.min(),
    ):
        with patch.object(HistoricalDownloader, "_download_ccxt", fake_download_ccxt):
            with patch.object(downloader, "_default_bar_count", return_value=10_000_000):
                df = downloader.load_or_download(n_bars=None)

    assert len(df) == 2003  # original cache + 3 freshly fetched bars


def test_context_window_reuses_late_listed_cache(tmp_path):
    """A context symbol that listed after the primary must not re-download every run.

    The cached context history starts later than the primary window (the symbol listed
    afterwards) but reaches the exchange's earliest candle and is current. Enrichment
    must reuse the cached slice instead of re-fetching the whole window.
    """
    config = AppConfig.model_validate({"data": {"data_dir": str(tmp_path / "data")}})
    downloader = HistoricalDownloader(config)
    # Primary window stretches back before the context symbol's first candle.
    window_start = pd.Timestamp("2019-09-08 18:00:00", tz="UTC")
    window_end = pd.Timestamp.now(tz="UTC").floor("15min")
    # Context cache begins well after the window start but is live to the present.
    cached = _ohlcv_frame(3000, live=True)
    cache_path = downloader._cache_path("ETH/USDT")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached.to_parquet(cache_path)

    with patch.object(
        HistoricalDownloader,
        "_exchange_earliest_ts",
        lambda self, sym: cached.index.min(),
    ):
        with patch.object(
            HistoricalDownloader,
            "_download_ccxt",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("must reuse cache")),
        ):
            out = downloader.load_or_download(
                "ETH/USDT",
                align_index=pd.date_range(window_start, window_end, freq="15min", tz="UTC"),
                skip_enrichment=True,
            )
    assert out is not None and not out.empty
    assert len(out) == 3000


def test_window_fetch_skips_when_listing_after_window(tmp_path, monkeypatch):
    """Context window fetch must not treat post-window listings as a successful download."""
    config = AppConfig.model_validate({"data": {"data_dir": str(tmp_path / "data")}})
    downloader = HistoricalDownloader(config)
    window_start = pd.Timestamp("2019-09-08 17:58:00", tz="UTC")
    window_end = pd.Timestamp("2019-09-12 05:17:00", tz="UTC")
    tf_ms = 15 * 60_000

    def fake_download_ccxt(self, symbol, target_bars, *, base_df=None, since_ts=None, until_ts=None, recent_tail=False):
        del symbol, target_bars, base_df, recent_tail
        start_ms = int(pd.Timestamp("2020-01-01", tz="UTC").timestamp() * 1000)
        rows = [[start_ms + i * tf_ms, 1.0, 2.0, 0.5, 1.5, 10.0] for i in range(1000)]
        return HistoricalDownloader._rows_to_dataframe(rows)

    monkeypatch.setattr(HistoricalDownloader, "_download_ccxt", fake_download_ccxt)
    out = downloader.load_or_download(
        "ETH/USDT",
        align_index=pd.date_range(window_start, window_end, freq="15min", tz="UTC"),
        skip_enrichment=True,
    )
    assert out is None or out.empty


def test_load_for_window_uses_cached_slice(tmp_path, monkeypatch):
    """Context joins request the primary timestamp window, not the latest tail."""
    config = AppConfig.model_validate({"data": {"data_dir": str(tmp_path / "data")}})
    downloader = HistoricalDownloader(config)
    window_start = pd.Timestamp("2019-09-08 17:58:00", tz="UTC")
    window_end = pd.Timestamp("2019-09-12 05:17:00", tz="UTC")
    cached = _ohlcv_frame(6000, start="2019-09-01")
    cache_path = downloader._cache_path("ETH/USDT")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached.to_parquet(cache_path)

    monkeypatch.setattr(
        HistoricalDownloader,
        "_download_ccxt",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should use cache")),
    )
    out = downloader.load_or_download(
        "ETH/USDT",
        align_index=pd.date_range(window_start, window_end, freq="1min", tz="UTC"),
        skip_enrichment=True,
    )
    assert out.index.min() <= window_start
    assert out.index.max() <= window_end
    assert out.index.max() >= window_end - pd.Timedelta(minutes=5)


def test_cache_only_skips_download_when_sufficient(tmp_path, monkeypatch):
    config = AppConfig.model_validate({"data": {"data_dir": str(tmp_path / "data")}})
    downloader = HistoricalDownloader(config)
    cached = _ohlcv_frame(1000, live=True)
    cache_path = downloader._cache_path(config.primary_symbol)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached.to_parquet(cache_path)

    monkeypatch.setattr(
        HistoricalDownloader,
        "_download_ccxt",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not download")),
    )
    df = downloader.load_or_download(n_bars=500, fetch_if_missing=False)
    assert len(df) == 500
    assert df.index.max() == cached.index.max()


def test_cache_only_raises_when_cache_too_small(tmp_path, monkeypatch):
    config = AppConfig.model_validate({"data": {"data_dir": str(tmp_path / "data")}})
    downloader = HistoricalDownloader(config)
    cached = _ohlcv_frame(200, live=True)
    cache_path = downloader._cache_path(config.primary_symbol)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached.to_parquet(cache_path)

    monkeypatch.setattr(
        HistoricalDownloader,
        "_download_ccxt",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not download")),
    )
    with pytest.raises(RuntimeError, match="Cache has 200 bars"):
        downloader.load_or_download(n_bars=500, fetch_if_missing=False)


def test_cache_only_raises_when_missing(tmp_path, monkeypatch):
    config = AppConfig.model_validate({"data": {"data_dir": str(tmp_path / "data")}})
    downloader = HistoricalDownloader(config)
    monkeypatch.setattr(
        HistoricalDownloader,
        "_download_ccxt",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not download")),
    )
    with pytest.raises(RuntimeError, match="No cached data"):
        downloader.load_or_download(n_bars=500, fetch_if_missing=False)


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

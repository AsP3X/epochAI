"""Tests for the Fear & Greed data source and its causal join in the downloader."""

from __future__ import annotations

import pandas as pd

from epoch_ai.config.settings import AppConfig
from epoch_ai.data import fear_greed as fng_mod
from epoch_ai.data.downloader import HistoricalDownloader


def _sample_payload() -> dict:
    # Three daily readings (epoch seconds, 00:00 UTC), newest-first like the real API.
    return {
        "data": [
            {"timestamp": "1577923200", "value": "30"},  # 2020-01-02
            {"timestamp": "1577836800", "value": "55"},  # 2020-01-01
            {"timestamp": "1578009600", "value": "70"},  # 2020-01-03
            {"timestamp": "bad", "value": "x"},  # malformed -> skipped
        ]
    }


def test_parse_fear_greed_sorts_and_skips_bad_records():
    series = fng_mod.parse_fear_greed(_sample_payload())
    assert series is not None
    assert list(series) == [55.0, 30.0, 70.0]  # chronological order
    assert series.index.is_monotonic_increasing
    assert str(series.index.tz) == "UTC"
    assert series.name == "fear_greed"


def test_parse_fear_greed_empty_returns_none():
    assert fng_mod.parse_fear_greed({"data": []}) is None
    assert fng_mod.parse_fear_greed({}) is None


def test_fetch_fear_greed_handles_network_error(monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("no network")

    monkeypatch.setattr(fng_mod, "urlopen", boom)
    assert fng_mod.fetch_fear_greed() is None


def test_attach_fear_greed_is_causal_forward_fill(tmp_path, monkeypatch):
    config = AppConfig.model_validate(
        {"data": {"data_dir": str(tmp_path / "data"), "include_fear_greed": True}}
    )
    downloader = HistoricalDownloader(config)

    # 15m bars spanning the sample daily readings.
    index = pd.date_range("2020-01-01", periods=400, freq="15min", tz="UTC")
    df = pd.DataFrame({"close": range(len(index))}, index=index, dtype=float)

    daily = pd.Series(
        [55.0, 30.0, 70.0],
        index=pd.to_datetime(
            ["2020-01-01", "2020-01-02", "2020-01-03"], utc=True
        ),
        name="fear_greed",
    )
    monkeypatch.setattr(
        "epoch_ai.data.fear_greed.fetch_fear_greed", lambda *a, **k: daily
    )

    out = downloader._attach_fear_greed(df)
    assert "fear_greed" in out.columns
    # First reading covers Jan 1; the value never reflects a *future* day's reading.
    assert out.loc["2020-01-01 00:00", "fear_greed"] == 55.0
    assert out.loc["2020-01-01 23:45", "fear_greed"] == 55.0  # still Jan 1's value
    assert out.loc["2020-01-02 06:00", "fear_greed"] == 30.0
    assert out.loc["2020-01-03 12:00", "fear_greed"] == 70.0
    # Causality guarantee: every bar equals the latest reading at/<= that bar.
    expected = daily.reindex(out.index, method="ffill")
    pd.testing.assert_series_equal(out["fear_greed"], expected, check_names=False)


def test_attach_fear_greed_graceful_when_unavailable(tmp_path, monkeypatch):
    config = AppConfig.model_validate(
        {"data": {"data_dir": str(tmp_path / "data"), "include_fear_greed": True}}
    )
    downloader = HistoricalDownloader(config)
    index = pd.date_range("2020-01-01", periods=50, freq="15min", tz="UTC")
    df = pd.DataFrame({"close": range(len(index))}, index=index, dtype=float)

    monkeypatch.setattr(
        "epoch_ai.data.fear_greed.fetch_fear_greed", lambda *a, **k: None
    )
    out = downloader._attach_fear_greed(df)
    assert "fear_greed" not in out.columns  # no column, pipeline still works


def test_download_joins_fear_greed_end_to_end(tmp_path, monkeypatch):
    """A full offline download with the toggle on yields a usable sentiment column."""
    config = AppConfig.model_validate(
        {
            "data": {
                "data_dir": str(tmp_path / "data"),
                "use_synthetic_fallback": True,
                "include_fear_greed": True,
            }
        }
    )
    monkeypatch.setattr(
        HistoricalDownloader, "_download_ccxt", lambda *a, **k: None
    )
    # Daily readings spanning the synthetic range (starts 2017-01-01 in earliest mode).
    daily = pd.Series(
        [40.0, 60.0],
        index=pd.to_datetime(["2016-12-01", "2017-02-01"], utc=True),
        name="fear_greed",
    )
    monkeypatch.setattr(
        "epoch_ai.data.fear_greed.fetch_fear_greed", lambda *a, **k: daily
    )

    df = HistoricalDownloader(config).load_or_download(n_bars=500)
    assert "fear_greed" in df.columns
    assert df["fear_greed"].notna().all()  # ffill+bfill leaves no gaps on the grid
    assert df["fear_greed"].nunique() >= 1

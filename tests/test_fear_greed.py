"""Tests for the Fear & Greed data source and its causal join in enrichment."""

from __future__ import annotations

import pandas as pd

from epoch_ai.config.settings import AppConfig
from epoch_ai.data import enrichment as enrich_mod
from epoch_ai.data import fear_greed as fng_mod
from epoch_ai.data.downloader import HistoricalDownloader
from epoch_ai.data.enrichment import enrich_primary_market


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


def test_join_fear_greed_is_causal_forward_fill(tmp_path, monkeypatch):
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
    # _join_fear_greed pulls the daily series via _load_fear_greed_series(cache).
    monkeypatch.setattr(enrich_mod, "_load_fear_greed_series", lambda cache: daily)

    out = enrich_mod._join_fear_greed(df, tmp_path)
    assert "fear_greed" in out.columns
    # First reading covers Jan 1; the value never reflects a *future* day's reading.
    assert out.loc["2020-01-01 00:00", "fear_greed"] == 55.0
    assert out.loc["2020-01-01 23:45", "fear_greed"] == 55.0  # still Jan 1's value
    assert out.loc["2020-01-02 06:00", "fear_greed"] == 30.0
    assert out.loc["2020-01-03 12:00", "fear_greed"] == 70.0
    # Causality guarantee: every bar equals the latest reading at/<= that bar.
    expected = daily.reindex(out.index, method="ffill")
    pd.testing.assert_series_equal(out["fear_greed"], expected, check_names=False)


def test_join_fear_greed_graceful_when_unavailable(tmp_path, monkeypatch):
    index = pd.date_range("2020-01-01", periods=50, freq="15min", tz="UTC")
    df = pd.DataFrame({"close": range(len(index))}, index=index, dtype=float)

    monkeypatch.setattr(enrich_mod, "_load_fear_greed_series", lambda cache: None)
    out = enrich_mod._join_fear_greed(df, tmp_path)
    assert "fear_greed" not in out.columns  # no column, pipeline still works


def test_enrich_joins_fear_greed_end_to_end(tmp_path, monkeypatch):
    """Enrichment with the toggle on yields a usable, causal sentiment column."""
    config = AppConfig.model_validate(
        {
            "data": {
                "data_dir": str(tmp_path / "data"),
                "context_symbols": [],
                "fetch_fear_greed": True,
                "fetch_spot_basis": False,
            }
        }
    )
    index = pd.date_range("2020-01-01", periods=300, freq="15min", tz="UTC")
    btc = pd.DataFrame({"close": range(len(index))}, index=index, dtype=float)

    daily = pd.Series(
        [40.0, 60.0],
        index=pd.to_datetime(["2020-01-01", "2020-01-02"], utc=True),
        name="fear_greed",
    )
    monkeypatch.setattr(enrich_mod, "_load_fear_greed_series", lambda cache: daily)

    downloader = HistoricalDownloader(config)
    enriched = enrich_primary_market(btc, config, downloader)
    assert "fear_greed" in enriched.columns
    assert enriched["fear_greed"].notna().all()  # readings start at the first bar
    assert set(enriched["fear_greed"].unique()) == {40.0, 60.0}

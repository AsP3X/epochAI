"""Tests for market enrichment (cross-asset, sentiment, basis joins)."""

from __future__ import annotations

import json
from unittest.mock import patch

import pandas as pd

from epoch_ai.config.settings import AppConfig
from epoch_ai.data.downloader import HistoricalDownloader
from epoch_ai.data.enrichment import enrich_primary_market


def _btc_frame(n: int = 200) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=n, freq="15min", tz="UTC")
    close = pd.Series(100.0 + pd.Series(range(n)).values, index=index)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 10.0,
            "funding_rate": 0.0001,
        },
        index=index,
    )


def test_joins_context_symbol_columns(tmp_path):
    config = AppConfig.model_validate(
        {
            "data": {
                "data_dir": str(tmp_path / "data"),
                "context_symbols": ["ETH/USDT"],
                "fetch_fear_greed": False,
                "fetch_spot_basis": False,
            }
        }
    )
    btc = _btc_frame(200)
    eth = _btc_frame(200)
    eth["close"] = eth["close"] * 0.04
    downloader = HistoricalDownloader(config)

    def fake_load(symbol, **kwargs):
        del kwargs
        return eth.copy() if symbol == "ETH/USDT" else btc.copy()

    with patch.object(downloader, "load_or_download", side_effect=fake_load):
        enriched = enrich_primary_market(btc, config, downloader)

    assert "eth_close" in enriched.columns
    assert "eth_volume" in enriched.columns
    assert enriched["eth_close"].iloc[-1] == eth["close"].iloc[-1]


def test_joins_fear_greed_from_api(tmp_path):
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
    btc = _btc_frame(100)
    payload = json.dumps(
        {
            "data": [
                {"timestamp": "1577836800", "value": "40"},
                {"timestamp": "1577923200", "value": "55"},
            ]
        }
    ).encode()

    class FakeResp:
        def read(self):
            return payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    downloader = HistoricalDownloader(config)
    with patch("epoch_ai.data.enrichment.urllib.request.urlopen", return_value=FakeResp()):
        enriched = enrich_primary_market(btc, config, downloader)

    assert "fear_greed" in enriched.columns
    assert enriched["fear_greed"].notna().any()

"""Tests for cross-asset context joins and the CrossAssetFeatures group.

The current architecture joins full context-symbol OHLCV(+derivatives) via
:func:`epoch_ai.data.enrichment.enrich_primary_market` (columns prefixed by
:func:`epoch_ai.data.symbols.asset_prefix`, e.g. ``eth_close``) and derives relative
signals in :class:`epoch_ai.features.cross_asset.CrossAssetFeatures`.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

from epoch_ai.config.settings import AppConfig
from epoch_ai.data.downloader import HistoricalDownloader
from epoch_ai.data.enrichment import enrich_primary_market
from epoch_ai.data.symbols import asset_prefix
from epoch_ai.features.cross_asset import CrossAssetFeatures
from epoch_ai.features.pipeline import FeaturePipeline


def _ohlcv(n: int, *, start: str, base: float) -> pd.DataFrame:
    index = pd.date_range(start=start, periods=n, freq="15min", tz="UTC")
    close = pd.Series(np.arange(n, dtype=float), index=index) + base
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1.0,
            "funding_rate": 0.0001,
        },
        index=index,
    )


def test_cross_asset_enabled_by_default():
    assert AppConfig().features.cross_asset is True
    assert "ETH/USDT" in AppConfig().data.context_symbols


def test_enrich_joins_context_columns_same_bar_causal(tmp_path):
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
    downloader = HistoricalDownloader(config)

    btc = _ohlcv(300, start="2020-01-01", base=10000.0)
    eth = _ohlcv(300, start="2020-01-01", base=200.0)

    def fake_load(symbol, **kwargs):
        del kwargs
        return eth.copy() if symbol == "ETH/USDT" else btc.copy()

    with patch.object(downloader, "load_or_download", side_effect=fake_load):
        out = enrich_primary_market(btc, config, downloader)

    col = f"{asset_prefix('ETH/USDT')}_close"
    assert col in out.columns
    # Same-bar alignment: each BTC bar carries the ETH close of the *same* timestamp.
    pd.testing.assert_series_equal(
        out[col], eth["close"].reindex(out.index), check_names=False
    )


def test_enrich_skips_primary_symbol(tmp_path):
    config = AppConfig.model_validate(
        {
            "data": {
                "data_dir": str(tmp_path / "data"),
                "context_symbols": ["BTC/USDT"],
                "fetch_fear_greed": False,
                "fetch_spot_basis": False,
            }
        }
    )
    downloader = HistoricalDownloader(config)
    btc = _ohlcv(50, start="2020-01-01", base=10000.0)
    # The only configured context symbol equals the primary symbol -> nothing to join.
    out = enrich_primary_market(btc, config, downloader)
    assert not any(c.startswith("btc_") for c in out.columns)


def test_cross_asset_feature_group_emits_when_columns_present():
    pfx = asset_prefix("ETH/USDT")
    btc = _ohlcv(400, start="2020-01-01", base=10000.0)
    btc[f"{pfx}_close"] = np.linspace(200.0, 400.0, len(btc))
    feats = CrossAssetFeatures(context_symbols=["ETH/USDT"]).compute(btc)
    assert any(c.startswith(f"xasset_{pfx}_ret_") for c in feats.columns)
    assert any("_corr_" in c for c in feats.columns)


def test_cross_asset_feature_group_noop_without_columns():
    btc = _ohlcv(100, start="2020-01-01", base=10000.0)
    feats = CrossAssetFeatures(context_symbols=["ETH/USDT"]).compute(btc)
    assert feats.shape[1] == 0  # graceful: no context prices -> no columns


def test_cross_asset_constant_region_does_not_produce_nan_corr():
    # A pre-listing region back-filled to a flat context price has zero return variance,
    # making the rolling correlation undefined. It must degrade to 0 (not NaN) so those
    # rows survive the pipeline dropna instead of being dropped wholesale.
    pfx = asset_prefix("ETH/USDT")
    btc = _ohlcv(400, start="2020-01-01", base=10000.0)
    flat_then_moving = np.concatenate(
        [np.full(200, 200.0), np.linspace(200.0, 400.0, 200)]
    )
    btc[f"{pfx}_close"] = flat_then_moving
    feats = CrossAssetFeatures(context_symbols=["ETH/USDT"]).compute(btc)
    corr_col = next(c for c in feats.columns if "_corr_" in c)
    # The constant front half yields corr == 0 (after warm-up), never NaN.
    constant_region = feats[corr_col].iloc[96:200]
    assert constant_region.notna().all()
    assert (constant_region == 0.0).all()


def test_pipeline_with_cross_asset(market, small_config):
    small_config.features.cross_asset = True
    pfx = asset_prefix("ETH/USDT")
    enriched = market.copy()
    enriched[f"{pfx}_close"] = np.linspace(200.0, 400.0, len(enriched))
    feats = FeaturePipeline(small_config).transform(enriched)
    assert any(c.startswith("xasset_") for c in feats.columns)

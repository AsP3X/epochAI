"""Tests for cross-asset correlating-data joins and the CrossAssetFeatures group."""

from __future__ import annotations

import numpy as np
import pandas as pd

from epoch_ai.config.settings import AppConfig
from epoch_ai.data.downloader import HistoricalDownloader
from epoch_ai.features.cross_asset import CrossAssetFeatures, cross_asset_column
from epoch_ai.features.pipeline import FeaturePipeline


def _ohlcv(n: int, *, start: str, base: float) -> pd.DataFrame:
    index = pd.date_range(start=start, periods=n, freq="15min", tz="UTC")
    close = pd.Series(np.arange(n, dtype=float), index=index) + base
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1.0},
        index=index,
    )


def test_cross_asset_disabled_by_default():
    assert AppConfig().features.cross_asset is False
    assert AppConfig().data.cross_asset_symbols == []


def test_attach_cross_assets_same_bar_causal(tmp_path, monkeypatch):
    config = AppConfig.model_validate(
        {"data": {"data_dir": str(tmp_path / "data"), "cross_asset_symbols": ["ETH/USDT"]}}
    )
    downloader = HistoricalDownloader(config)

    btc = _ohlcv(300, start="2020-01-01", base=10000.0)
    eth = _ohlcv(300, start="2020-01-01", base=200.0)

    # Reference loads return the ETH frame regardless of which sibling downloader runs.
    monkeypatch.setattr(
        HistoricalDownloader,
        "load_or_download",
        lambda self, symbol=None, *, n_bars=None, force=False: eth,
    )

    out = downloader._attach_cross_assets(btc, "BTC/USDT")
    col = cross_asset_column("ETH/USDT")
    assert col in out.columns
    # Same-bar alignment: each BTC bar carries the ETH close of the *same* timestamp.
    pd.testing.assert_series_equal(
        out[col], eth["close"].reindex(out.index), check_names=False
    )


def test_attach_cross_assets_skips_primary_symbol(tmp_path):
    config = AppConfig.model_validate(
        {"data": {"data_dir": str(tmp_path / "data"), "cross_asset_symbols": ["BTC/USDT"]}}
    )
    downloader = HistoricalDownloader(config)
    btc = _ohlcv(50, start="2020-01-01", base=10000.0)
    # The only configured symbol equals the primary symbol -> nothing to join, no load.
    out = downloader._attach_cross_assets(btc, "BTC/USDT")
    assert not any(c.startswith("xa_") for c in out.columns)


def test_cross_asset_feature_group_emits_when_columns_present():
    btc = _ohlcv(400, start="2020-01-01", base=10000.0)
    btc[cross_asset_column("ETH/USDT")] = np.linspace(200.0, 400.0, len(btc))
    feats = CrossAssetFeatures().compute(btc)
    assert any(c.endswith("_relstr") for c in feats.columns)
    assert any("_corr_" in c for c in feats.columns)


def test_cross_asset_feature_group_noop_without_columns():
    btc = _ohlcv(100, start="2020-01-01", base=10000.0)
    feats = CrossAssetFeatures().compute(btc)
    assert feats.shape[1] == 0  # graceful: no cross prices -> no columns


def test_cross_asset_constant_region_does_not_produce_nan_corr():
    # A pre-listing region back-filled to a flat cross price has zero return variance,
    # making the rolling correlation undefined. It must degrade to 0 (not NaN) so those
    # rows survive the pipeline dropna instead of being dropped wholesale.
    btc = _ohlcv(400, start="2020-01-01", base=10000.0)
    col = cross_asset_column("ETH/USDT")
    flat_then_moving = np.concatenate(
        [np.full(200, 200.0), np.linspace(200.0, 400.0, 200)]
    )
    btc[col] = flat_then_moving
    feats = CrossAssetFeatures().compute(btc)
    corr_col = next(c for c in feats.columns if "_corr_" in c)
    # The constant front half yields corr == 0 (after warm-up), never NaN.
    constant_region = feats[corr_col].iloc[96:200]
    assert constant_region.notna().all()
    assert (constant_region == 0.0).all()


def test_pipeline_with_cross_asset(market, small_config):
    small_config.features.cross_asset = True
    enriched = market.copy()
    enriched[cross_asset_column("ETH/USDT")] = np.linspace(
        200.0, 400.0, len(enriched)
    )
    feats = FeaturePipeline(small_config).transform(enriched)
    assert any(c.startswith("xa_") for c in feats.columns)

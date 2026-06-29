"""Tests for the synthetic data generator and cleaning."""

from __future__ import annotations

import pandas as pd

from epoch_ai.data.cleaning import align_and_clean
from epoch_ai.data.synthetic import generate_synthetic_ohlcv


def test_shape_and_columns(market):
    assert len(market) == 4000
    for col in ["open", "high", "low", "close", "volume", "funding_rate", "open_interest"]:
        assert col in market.columns


def test_ohlc_consistency(market):
    assert (market["high"] >= market["low"]).all()
    assert (market["high"] >= market["close"]).all()
    assert (market["low"] <= market["close"]).all()
    assert (market["close"] > 0).all()


def test_reproducible():
    a = generate_synthetic_ohlcv(timeframe="15m", start="2020-01-01", n_bars=500, seed=3)
    b = generate_synthetic_ohlcv(timeframe="15m", start="2020-01-01", n_bars=500, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_cleaning_fills_gaps(market):
    gapped = market.drop(market.index[100:110])
    cleaned = align_and_clean(gapped, "15m")
    assert len(cleaned) == len(market)
    assert not cleaned.isna().any().any()

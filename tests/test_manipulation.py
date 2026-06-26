"""Tests for manipulation proxy features."""

from __future__ import annotations

from epoch_ai.features.manipulation import ManipulationFeatures


def test_manipulation_columns_present(market):
    out = ManipulationFeatures().compute(market)
    expected = [
        "manip_vol_price_div",
        "manip_wick_cluster",
        "manip_illiq_spike",
        "manip_return_skew",
        "manip_gap_recovery",
    ]
    for col in expected:
        assert col in out.columns


def test_manipulation_uses_derivatives_when_present(market):
    df = market.copy()
    df["open_interest"] = df["close"] * 100
    df["funding_rate"] = 0.0001
    out = ManipulationFeatures().compute(df)
    assert "manip_oi_price_div" in out.columns
    assert "manip_funding_extreme" in out.columns

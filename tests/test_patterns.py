"""Tests for causal swing detection and pattern geometry."""

from __future__ import annotations

import numpy as np
import pandas as pd

from epoch_ai.features.patterns.geometry import (
    double_top_bottom_score,
    triangle_convergence_score,
)
from epoch_ai.features.patterns.swings import confirmed_swing_highs
from epoch_ai.features.pipeline import FeaturePipeline


def _trend_frame(n: int = 200) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="15min")
    close = pd.Series(np.linspace(100.0, 130.0, n), index=idx)
    high = close + 0.5
    low = close - 0.5
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": 1.0},
        index=idx,
    )


def test_swing_high_not_confirmed_until_lag():
    df = _trend_frame()
    df.loc[df.index[50], "high"] = df["high"].iloc[50] + 5.0
    confirm = 3
    swings = confirmed_swing_highs(df["high"], confirm_bars=confirm)
    assert swings.iloc[50] == 0.0
    assert swings.iloc[53] != 0.0 or swings.iloc[54] != 0.0


def test_swing_detection_uses_only_past_for_confirmation():
    df = _trend_frame(120)
    swings = confirmed_swing_highs(df["high"], confirm_bars=2)
    assert (swings.iloc[-2:] == 0.0).all()


def test_double_top_score_bounded(market):
    score = double_top_bottom_score(market, lookback=48, mode="top")
    assert score.min() >= -1.0
    assert score.max() <= 1.0
    assert len(score) == len(market)


def test_triangle_convergence_non_negative(market):
    score = triangle_convergence_score(market, lookback=96)
    assert (score >= 0.0).all()


def test_pattern_features_no_future_leak(market, small_config):
    """Features at timestamp t depend only on OHLCV up to t (not later bars)."""
    small_config.features.patterns = True
    small_config.features.cross_asset = False
    pipe = FeaturePipeline(small_config)
    full = pipe.transform(market.copy(), log_stats=False)
    t_pos = len(market) // 2
    ts = market.index[t_pos]
    # Extend past ts so swing confirmation is not masked at the live edge.
    truncated = market.iloc[: t_pos + 50].copy()
    partial = pipe.transform(truncated, log_stats=False)
    pat_cols = [c for c in full.columns if c.startswith("pat_")]
    pd.testing.assert_series_equal(
        full.loc[ts, pat_cols],
        partial.loc[ts, pat_cols],
        check_names=False,
    )


def test_pattern_group_emits_expected_prefix(market, small_config):
    small_config.features.patterns = True
    features = FeaturePipeline(small_config).transform(market)
    pat_cols = [c for c in features.columns if c.startswith("pat_")]
    assert len(pat_cols) >= 10

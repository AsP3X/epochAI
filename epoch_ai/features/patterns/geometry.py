"""Continuous chart-pattern geometry scores (causal, window ends at current bar)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def double_top_bottom_score(
    df: pd.DataFrame,
    lookback: int,
    *,
    mode: str = "top",
    pivot_confirm_bars: int = 3,
) -> pd.Series:
    """Score similarity of two extrema within a rolling window."""
    col = "high" if mode == "top" else "low"
    series = df[col]
    min_p = max(2, lookback // 2)
    roll = series.rolling(lookback, min_periods=min_p)
    if mode == "top":
        primary = roll.max()
        secondary = (
            series.shift(pivot_confirm_bars + 1)
            .rolling(lookback, min_periods=min_p)
            .max()
        )
    else:
        primary = roll.min()
        secondary = (
            series.shift(pivot_confirm_bars + 1)
            .rolling(lookback, min_periods=min_p)
            .min()
        )
    similarity = 1.0 - (primary - secondary).abs() / series.replace(0.0, np.nan)
    return similarity.fillna(0.0).clip(0.0, 1.0)


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Causal rolling OLS slope (x = 0..window-1)."""
    min_p = max(2, window // 2)

    def _slope(arr: np.ndarray) -> float:
        if len(arr) < 2:
            return 0.0
        x = np.arange(len(arr), dtype=float)
        x = x - x.mean()
        y = arr - arr.mean()
        denom = (x * x).sum()
        if denom == 0.0:
            return 0.0
        return float((x * y).sum() / denom)

    return series.rolling(window, min_periods=min_p).apply(_slope, raw=True)


def triangle_convergence_score(df: pd.DataFrame, lookback: int) -> pd.Series:
    """Higher when high and low boundary slopes converge (triangle narrowing)."""
    high_slope = _rolling_slope(df["high"], lookback)
    low_slope = _rolling_slope(df["low"], lookback)
    convergence = (high_slope - low_slope).abs() / df["close"].replace(0.0, np.nan)
    inv = 1.0 / (1.0 + convergence)
    return inv.fillna(0.0).clip(0.0, 1.0)


def flag_pole_score(df: pd.DataFrame, lookback: int) -> pd.Series:
    """Impulse move in first half vs tight range in second half of window."""
    min_p = max(2, lookback // 2)
    close = df["close"]
    ret = close.pct_change()
    pole = ret.rolling(lookback // 2, min_periods=min_p // 2).sum().abs()
    rng = (df["high"] - df["low"]) / close.replace(0.0, np.nan)
    flag_tight = 1.0 - rng.rolling(lookback // 2, min_periods=min_p // 2).mean()
    score = pole * flag_tight.clip(0.0, 1.0)
    return score.fillna(0.0).clip(0.0, 1.0)


def wedge_score(df: pd.DataFrame, lookback: int) -> pd.Series:
    """Both boundaries slope same direction while range narrows."""
    high_slope = _rolling_slope(df["high"], lookback)
    low_slope = _rolling_slope(df["low"], lookback)
    same_dir = np.sign(high_slope) == np.sign(low_slope)
    rng = (df["high"].rolling(lookback, min_periods=lookback // 2).max()
           - df["low"].rolling(lookback, min_periods=lookback // 2).min())
    rng_norm = rng / df["close"].replace(0.0, np.nan)
    narrow = 1.0 - rng_norm / rng_norm.rolling(lookback, min_periods=lookback // 2).max().replace(
        0.0, np.nan
    )
    score = same_dir.astype(float) * narrow
    return score.fillna(0.0).clip(0.0, 1.0)


def breakout_strength(df: pd.DataFrame, lookback: int) -> pd.Series:
    """Close beyond rolling range scaled by volume z-score."""
    min_p = max(2, lookback // 2)
    close = df["close"]
    roll_high = df["high"].rolling(lookback, min_periods=min_p).max().shift(1)
    roll_low = df["low"].rolling(lookback, min_periods=min_p).min().shift(1)
    up_break = (close - roll_high) / close.replace(0.0, np.nan)
    down_break = (roll_low - close) / close.replace(0.0, np.nan)
    raw = up_break.clip(lower=0.0) + down_break.clip(lower=0.0)
    vol = df.get("volume")
    if vol is not None:
        vol_z = (vol - vol.rolling(lookback, min_periods=min_p).mean()) / vol.rolling(
            lookback, min_periods=min_p
        ).std().replace(0.0, np.nan)
        raw = raw * (1.0 + vol_z.fillna(0.0).clip(-1.0, 3.0))
    return raw.fillna(0.0).clip(0.0, 1.0)


def head_shoulders_score(
    df: pd.DataFrame,
    lookback: int,
    pivot_confirm_bars: int = 3,
) -> pd.DataFrame:
    """Symmetry proxy for head-and-shoulders and inverse variants."""
    third = max(2, lookback // 3)
    high = df["high"]
    low = df["low"]
    left_h = high.rolling(third, min_periods=third).max()
    mid_h = high.shift(third).rolling(third, min_periods=third).max()
    right_h = high.shift(2 * third).rolling(third, min_periods=third).max()
    shoulder_avg = (left_h + right_h) / 2.0
    top = ((mid_h - shoulder_avg) / high.replace(0.0, np.nan)).clip(0.0, 1.0)

    left_l = low.rolling(third, min_periods=third).min()
    mid_l = low.shift(third).rolling(third, min_periods=third).min()
    right_l = low.shift(2 * third).rolling(third, min_periods=third).min()
    shoulder_low = (left_l + right_l) / 2.0
    inv = ((shoulder_low - mid_l) / low.replace(0.0, np.nan)).clip(0.0, 1.0)

    _ = pivot_confirm_bars  # reserved for future pivot-aware H&S refinement
    return pd.DataFrame({"top": top.fillna(0.0), "inv": inv.fillna(0.0)}, index=df.index)


def candlestick_context_score(df: pd.DataFrame) -> pd.DataFrame:
    """Engulfing strength and doji-at-extreme context."""
    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    close = df["close"]
    body = (close - open_).abs()
    rng = (high - low).replace(0.0, np.nan)
    body_ratio = body / rng
    prev_body = body.shift(1)
    engulf = (body / prev_body.replace(0.0, np.nan)).clip(0.0, 5.0) / 5.0
    direction_flip = (np.sign(close - open_) != np.sign(open_ - close.shift(1))).astype(float)
    engulf = engulf * (0.5 + 0.5 * direction_flip)

    doji = (1.0 - body_ratio).clip(0.0, 1.0)
    at_high = (close - low) / rng
    at_low = (high - close) / rng
    doji_ext = doji * at_high.combine(at_low, max)
    return pd.DataFrame(
        {"engulf": engulf.fillna(0.0), "doji_ext": doji_ext.fillna(0.0)},
        index=df.index,
    )

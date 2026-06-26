"""Causal swing pivot detection for pattern geometry."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _confirm_swing_events(
    series: pd.Series,
    *,
    confirm_bars: int,
    left: int,
    mode: str,
) -> pd.Series:
    """Mark bars where a pivot is confirmed (causal lag applied)."""
    n = len(series)
    arr = series.to_numpy(dtype=float)
    events = np.zeros(n, dtype=float)
    pivot_prices = np.full(n, np.nan)

    for i in range(left, n - confirm_bars):
        window = arr[i - left : i + confirm_bars + 1]
        center = arr[i]
        if mode == "high":
            is_pivot = center == np.nanmax(window) and np.sum(window == center) == 1
        else:
            is_pivot = center == np.nanmin(window) and np.sum(window == center) == 1
        if is_pivot:
            confirm_idx = i + confirm_bars
            events[confirm_idx] = 1.0
            pivot_prices[confirm_idx] = center

    return pd.Series(events, index=series.index), pd.Series(pivot_prices, index=series.index)


def confirmed_swing_highs(high: pd.Series, confirm_bars: int = 3, left: int = 2) -> pd.Series:
    """Return normalized distance from price to last confirmed swing high."""
    events, pivot_prices = _confirm_swing_events(
        high, confirm_bars=confirm_bars, left=left, mode="high"
    )
    swing_price = pivot_prices.where(events == 1.0).ffill()
    dist = (high - swing_price) / high.replace(0.0, np.nan)
    dist = dist.where(events != 1.0, 0.0)
    if confirm_bars > 0 and len(dist) > 0:
        dist.iloc[-confirm_bars:] = 0.0
    return dist.fillna(0.0).clip(-1.0, 1.0)


def confirmed_swing_lows(low: pd.Series, confirm_bars: int = 3, left: int = 2) -> pd.Series:
    """Return normalized distance from price to last confirmed swing low."""
    events, pivot_prices = _confirm_swing_events(
        low, confirm_bars=confirm_bars, left=left, mode="low"
    )
    swing_price = pivot_prices.where(events == 1.0).ffill()
    dist = (low - swing_price) / low.replace(0.0, np.nan)
    dist = dist.where(events != 1.0, 0.0)
    if confirm_bars > 0 and len(dist) > 0:
        dist.iloc[-confirm_bars:] = 0.0
    return dist.fillna(0.0).clip(-1.0, 1.0)

"""Shared causal rolling statistics for feature groups."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_z(series: pd.Series, window: int = 96, min_periods: int = 16) -> pd.Series:
    """Causal rolling z-score; flat windows degrade to neutral 0."""
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    z = (series - mean) / std.where(std > 0)
    return z.mask(std.eq(0.0), 0.0)


def rolling_zscore_frame(
    series: pd.Series,
    window: int,
    *,
    min_periods: int | None = None,
    prefix: str = "",
) -> pd.DataFrame:
    """Return level + z columns for a series."""
    mp = min_periods if min_periods is not None else max(4, window // 4)
    out = pd.DataFrame(index=series.index)
    if prefix:
        out[prefix] = series
        out[f"{prefix}_z"] = rolling_z(series, window, mp)
    return out


def pct_change_safe(series: pd.Series, periods: int = 1) -> pd.Series:
    return series.pct_change(periods, fill_method=None)


def signed_streak(close: pd.Series) -> pd.Series:
    """Consecutive up/down close streak length (signed)."""
    sign = np.sign(close.diff()).fillna(0.0)
    grp = (sign != sign.shift()).cumsum()
    return sign.groupby(grp).cumsum()

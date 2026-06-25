"""Data alignment and cleaning utilities."""

from __future__ import annotations

import pandas as pd

from epoch_ai.utils.logging import get_logger
from epoch_ai.utils.timeframe import timeframe_to_pandas_freq

logger = get_logger(__name__)

_OHLCV = ["open", "high", "low", "close", "volume"]


def align_and_clean(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Align an OHLCV(+context) frame to a regular grid and clean it.

    Steps:
        1. Ensure a sorted, unique, tz-aware ``timestamp`` index.
        2. Reindex onto a complete, regular time grid for the timeframe.
        3. Forward-fill context columns and OHLC; back-fill any leading gaps.
        4. Drop rows that are still entirely missing.

    Args:
        df: Raw OHLCV frame indexed by ``timestamp``.
        timeframe: Bar size used to build the regular grid.

    Returns:
        A cleaned, gap-free DataFrame on a regular grid.
    """
    if df.empty:
        return df

    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        raise TypeError("Expected a DatetimeIndex named 'timestamp'.")
    out = out[~out.index.duplicated(keep="last")].sort_index()

    full_index = pd.date_range(
        start=out.index[0],
        end=out.index[-1],
        freq=timeframe_to_pandas_freq(timeframe),
        tz=out.index.tz,
    )
    n_missing = len(full_index) - len(out)
    if n_missing > 0:
        logger.info("Reindexing: filling %d missing bars on the regular grid.", n_missing)
    out = out.reindex(full_index)
    out.index.name = "timestamp"

    # Carry the last known value forward across gaps; back-fill any leading NaNs.
    out = out.ffill().bfill()

    # Repair OHLC consistency (high>=max(o,c), low<=min(o,c)).
    if set(_OHLCV).issubset(out.columns):
        out["high"] = out[["high", "open", "close"]].max(axis=1)
        out["low"] = out[["low", "open", "close"]].min(axis=1)

    return out.dropna(how="all")

"""Higher-timeframe context features resampled from the bar grid.

Computes indicators on coarser candles (e.g. 1h, 4h), shifts by one HTF bar so
only *completed* higher-timeframe information is visible at each 15m row, then
forward-fills onto the primary index. Aligns engineered features with multi-bar
prediction horizons without look-ahead leakage.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from epoch_ai.features.base import FeatureGroup
from epoch_ai.features.technical import _adx, _rsi
from epoch_ai.utils.timeframe import timeframe_to_pandas_freq


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    o = df["open"].resample(rule).first()
    h = df["high"].resample(rule).max()
    low = df["low"].resample(rule).min()
    c = df["close"].resample(rule).last()
    v = df["volume"].resample(rule).sum() if "volume" in df.columns else None
    out = pd.DataFrame({"open": o, "high": h, "low": low, "close": c})
    if v is not None:
        out["volume"] = v
    return out.dropna(subset=["close"])


def _htf_block(htf: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Compute a compact HTF feature block; index is HTF bars."""
    close = htf["close"]
    ret = close.pct_change(fill_method=None)
    out = pd.DataFrame(index=htf.index)
    out[f"{prefix}_ret_1"] = ret
    out[f"{prefix}_ret_4"] = close.pct_change(4, fill_method=None)
    out[f"{prefix}_ret_6"] = close.pct_change(6, fill_method=None)
    out[f"{prefix}_rsi_14"] = _rsi(close, 14) / 100.0
    ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    out[f"{prefix}_ema_dist_50"] = close / ema50 - 1.0
    sma200 = close.rolling(200, min_periods=50).mean()
    out[f"{prefix}_sma_dist_200"] = close / sma200 - 1.0
    out[f"{prefix}_adx_14"] = _adx(htf, 14) / 100.0
    vol_s = ret.rolling(24, min_periods=12).std()
    vol_l = ret.rolling(96, min_periods=32).std()
    out[f"{prefix}_vol_ratio"] = vol_s / vol_l.replace(0.0, np.nan)
    out[f"{prefix}_trend_regime"] = np.sign(out[f"{prefix}_ema_dist_50"])
    return out


class HigherTimeframeFeatures(FeatureGroup):
    """Multi-timeframe trend and volatility context (causal shift + ffill)."""

    name = "htf"

    def __init__(self, htf_timeframes: Sequence[str] = ("1h", "4h")) -> None:
        self.htf_timeframes = tuple(htf_timeframes)

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        trend_cols: list[str] = []

        for tf in self.htf_timeframes:
            rule = timeframe_to_pandas_freq(tf)
            prefix = f"htf_{tf.replace(' ', '')}"
            htf = _resample_ohlcv(df, rule)
            if len(htf) < 10:
                continue
            block = _htf_block(htf, prefix)
            # Human: shift(1) ensures the 15m bar only sees the last *closed* HTF candle.
            # Agent: CAUSAL completed-bar only; ffill aligns to primary grid.
            aligned = block.shift(1).reindex(df.index, method="ffill")
            out = pd.concat([out, aligned], axis=1)
            trend_cols.append(f"{prefix}_trend_regime")

        if len(trend_cols) >= 2:
            a, b = trend_cols[0], trend_cols[1]
            out["htf_1h_4h_alignment"] = (out[a] == out[b]).astype(float)
        return out

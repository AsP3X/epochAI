"""Classic technical-analysis indicators.

Implemented in pure pandas/numpy so the system has **no hard dependency** on the
notoriously fragile ``pandas_ta`` package. (If ``pandas_ta`` is installed it can be
used to extend this group, but it is never required.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from epoch_ai.features.base import FeatureGroup


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


class TechnicalFeatures(FeatureGroup):
    """Trend, momentum and oscillator indicators."""

    name = "ta"

    def __init__(
        self,
        ma_windows: tuple[int, ...] = (10, 20, 50, 100, 200),
        rsi_periods: tuple[int, ...] = (7, 14, 28),
    ) -> None:
        self.ma_windows = ma_windows
        self.rsi_periods = rsi_periods

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        out = pd.DataFrame(index=df.index)

        # Returns over multiple lookbacks.
        for lag in (1, 3, 6, 12, 24, 48):
            out[f"ta_ret_{lag}"] = close.pct_change(lag)

        # Moving averages + price distance to them (scale-free).
        for w in self.ma_windows:
            sma = close.rolling(w, min_periods=w).mean()
            ema = close.ewm(span=w, adjust=False, min_periods=w).mean()
            out[f"ta_sma_dist_{w}"] = close / sma - 1.0
            out[f"ta_ema_dist_{w}"] = close / ema - 1.0

        # MA cross signals.
        fast = close.ewm(span=12, adjust=False, min_periods=12).mean()
        slow = close.ewm(span=26, adjust=False, min_periods=26).mean()
        macd = fast - slow
        signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
        out["ta_macd"] = macd / close
        out["ta_macd_signal"] = signal / close
        out["ta_macd_hist"] = (macd - signal) / close

        # RSI family.
        for p in self.rsi_periods:
            out[f"ta_rsi_{p}"] = _rsi(close, p)

        # ATR (normalised) + Bollinger position.
        atr = _atr(df, 14)
        out["ta_atr_pct"] = atr / close
        mid = close.rolling(20, min_periods=20).mean()
        std = close.rolling(20, min_periods=20).std()
        out["ta_bb_pos"] = (close - mid) / (2.0 * std).replace(0.0, np.nan)

        # Stochastic %K.
        low_n = df["low"].rolling(14, min_periods=14).min()
        high_n = df["high"].rolling(14, min_periods=14).max()
        out["ta_stoch_k"] = (close - low_n) / (high_n - low_n).replace(0.0, np.nan)

        return out

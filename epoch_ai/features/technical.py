"""Classic technical-analysis indicators.

Implemented in pure pandas/numpy so the system has **no hard dependency** on the
notoriously fragile ``pandas_ta`` package. (If ``pandas_ta`` is installed it can be
used to extend this group, but it is never required.)

All look-back windows are passed in from :class:`~epoch_ai.config.settings.FeatureConfig`
so the indicator set is config-driven rather than hard-coded. Every column is causal:
it uses only data available up to and including the current bar.
"""

from __future__ import annotations

from collections.abc import Sequence

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


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    return _true_range(df).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's Average Directional Index in [0, 100] (trend-strength, direction-free)."""
    up = df["high"].diff()
    down = -df["low"].diff()
    # Directional movement: only the dominant side counts on each bar.
    plus_dm = np.where((up > down) & (up > 0.0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0.0), down, 0.0)
    atr = _true_range(df).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1.0 / period, adjust=False, min_periods=period
    ).mean() / atr.replace(0.0, np.nan)
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1.0 / period, adjust=False, min_periods=period
    ).mean() / atr.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


class TechnicalFeatures(FeatureGroup):
    """Trend, momentum and oscillator indicators."""

    name = "ta"

    def __init__(
        self,
        return_lags: Sequence[int] = (1, 3, 6, 12, 24, 48),
        ma_windows: Sequence[int] = (10, 20, 50, 100, 200),
        rsi_periods: Sequence[int] = (7, 14, 28),
        vwap_window: int = 96,
    ) -> None:
        self.return_lags = tuple(return_lags)
        self.ma_windows = tuple(ma_windows)
        self.rsi_periods = tuple(rsi_periods)
        self.vwap_window = vwap_window

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        out = pd.DataFrame(index=df.index)

        # Returns over multiple lookbacks.
        for lag in self.return_lags:
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

        # --- Additional indicators (trend strength, mean-reversion, volume) ------
        # ADX: how strong is the trend (regardless of direction).
        out["ta_adx_14"] = _adx(df, 14) / 100.0

        # Williams %R(14): overbought/oversold oscillator in [-1, 0].
        out["ta_williams_r"] = (close - high_n) / (high_n - low_n).replace(0.0, np.nan)

        # CCI(20): deviation of typical price from its rolling mean.
        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        tp_sma = tp.rolling(20, min_periods=20).mean()
        mad = (tp - tp_sma).abs().rolling(20, min_periods=20).mean()
        out["ta_cci_20"] = (tp - tp_sma) / (0.015 * mad).replace(0.0, np.nan)

        # Rolling VWAP distance: price vs volume-weighted average over the window.
        vol = df.get("volume")
        if vol is not None:
            pv = (tp * vol).rolling(self.vwap_window, min_periods=self.vwap_window // 2).sum()
            vsum = vol.rolling(self.vwap_window, min_periods=self.vwap_window // 2).sum()
            vwap = pv / vsum.replace(0.0, np.nan)
            out["ta_vwap_dist"] = close / vwap - 1.0

            # OBV momentum: signed cumulative volume, z-scored to stay stationary.
            direction = np.sign(close.diff()).fillna(0.0)
            obv = (direction * vol).cumsum()
            obv_mean = obv.rolling(self.vwap_window, min_periods=self.vwap_window // 2).mean()
            obv_std = obv.rolling(
                self.vwap_window, min_periods=self.vwap_window // 2
            ).std().replace(0.0, np.nan)
            out["ta_obv_z"] = (obv - obv_mean) / obv_std

        return out

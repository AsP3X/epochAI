"""Classic technical-analysis indicators.

Implemented in pure pandas/numpy so the system has **no hard dependency** on the
notoriously fragile ``pandas_ta`` / ``pandas-ta`` package. (If installed it can be
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


def _directional_indicators(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series]:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0.0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0.0), down, 0.0)
    atr = _true_range(df).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1.0 / period, adjust=False, min_periods=period
    ).mean() / atr.replace(0.0, np.nan)
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1.0 / period, adjust=False, min_periods=period
    ).mean() / atr.replace(0.0, np.nan)
    return plus_di, minus_di


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

        # --- Extended momentum, trend quality, and channel position ----------------
        ret1 = close.pct_change(fill_method=None)
        out["ta_ret_accel_12"] = close.pct_change(12, fill_method=None) - close.pct_change(
            12, fill_method=None
        ).shift(12)
        out["ta_ret_accel_24"] = close.pct_change(24, fill_method=None) - close.pct_change(
            24, fill_method=None
        ).shift(24)
        for w in (48, 96):
            rsum = ret1.rolling(w, min_periods=w // 2).sum()
            rstd = ret1.rolling(w, min_periods=w // 2).std().replace(0.0, np.nan)
            out[f"ta_momentum_quality_{w}"] = rsum / rstd

        for w in (48,):
            path = close.diff().abs().rolling(w, min_periods=w // 2).sum()
            net = (close - close.shift(w)).abs()
            out[f"ta_efficiency_ratio_{w}"] = net / path.replace(0.0, np.nan)

        plus_di, minus_di = _directional_indicators(df, 14)
        out["ta_plus_di_14"] = plus_di / 100.0
        out["ta_minus_di_14"] = minus_di / 100.0
        out["ta_di_spread_14"] = (plus_di - minus_di) / 100.0

        for w in (12, 24):
            out[f"ta_roc_{w}"] = close.pct_change(w, fill_method=None)

        for w in (20, 55):
            hi = df["high"].rolling(w, min_periods=w // 2).max()
            lo = df["low"].rolling(w, min_periods=w // 2).min()
            out[f"ta_donchian_pos_{w}"] = (close - lo) / (hi - lo).replace(0.0, np.nan)

        ema20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
        atr14 = _atr(df, 14)
        out["ta_keltner_pos_20"] = (close - ema20) / (2.0 * atr14).replace(0.0, np.nan)

        span9 = close.ewm(span=9, adjust=False, min_periods=9).mean()
        span26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
        cloud_top = pd.concat([span9, span26], axis=1).max(axis=1)
        cloud_bot = pd.concat([span9, span26], axis=1).min(axis=1)
        out["ta_ichimoku_cloud_dist"] = np.where(
            close > cloud_top,
            (close - cloud_top) / close,
            np.where(close < cloud_bot, (close - cloud_bot) / close, 0.0),
        )

        hl2 = (df["high"] + df["low"]) / 2.0
        st_atr = _atr(df, 10)
        upper = hl2 + 3.0 * st_atr
        lower = hl2 - 3.0 * st_atr
        out["ta_supertrend_dist"] = np.where(
            close >= hl2, (close - lower) / close, (close - upper) / close
        )

        sma10 = close.rolling(10, min_periods=10).mean()
        sma50 = close.rolling(50, min_periods=50).mean()
        sma20 = close.rolling(20, min_periods=20).mean()
        sma200 = close.rolling(200, min_periods=200).mean()
        out["ta_ma_cross_10_50"] = np.sign(sma10 - sma50)
        out["ta_ma_cross_20_200"] = np.sign(sma20 - sma200)

        rsi14 = out.get("ta_rsi_14", _rsi(close, 14))
        out["ta_rsi_divergence_proxy_14"] = rsi14.diff(14) - ret1.rolling(14).sum()

        if vol is not None:
            vol_ma48 = vol.rolling(48, min_periods=12).mean()
            out["ta_volume_ma_ratio_48"] = vol / vol_ma48.replace(0.0, np.nan)
            pvt = (ret1 * vol).cumsum()
            pvt_mean = pvt.rolling(48, min_periods=12).mean()
            pvt_std = pvt.rolling(48, min_periods=12).std().replace(0.0, np.nan)
            out["ta_price_volume_trend_48"] = (pvt - pvt_mean) / pvt_std

        return out

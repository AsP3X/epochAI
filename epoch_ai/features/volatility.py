"""Volatility and market-regime features."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from epoch_ai.features.base import FeatureGroup


class VolatilityFeatures(FeatureGroup):
    """Realised volatility, vol-of-vol and simple regime descriptors."""

    name = "vol"

    def __init__(self, vol_windows: Sequence[int] = (12, 24, 48, 96)) -> None:
        self.vol_windows = tuple(vol_windows)

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        ret = df["close"].pct_change(fill_method=None)
        close = df["close"]

        for w in self.vol_windows:
            out[f"vol_std_{w}"] = ret.rolling(w, min_periods=max(2, w // 2)).std()

        vol_short = ret.rolling(24, min_periods=12).std()
        vol_long = ret.rolling(96, min_periods=32).std()
        out["vol_ratio"] = vol_short / vol_long.replace(0.0, np.nan)
        out["vol_of_vol"] = vol_short.rolling(48, min_periods=12).std()

        hl = np.log(df["high"] / df["low"]).replace([np.inf, -np.inf], np.nan)
        out["vol_parkinson"] = (hl.pow(2) / (4.0 * np.log(2.0))).rolling(
            24, min_periods=12
        ).mean().pow(0.5)

        out["vol_trend_strength"] = ret.rolling(48, min_periods=24).mean() / vol_long.replace(
            0.0, np.nan
        )

        roll_max = close.rolling(96, min_periods=24).max()
        out["vol_drawdown"] = close / roll_max - 1.0

        # Extended estimators and regime descriptors
        log_hl = np.log(df["high"] / df["low"]).replace([np.inf, -np.inf], np.nan)
        log_co = np.log(df["close"] / df["open"]).replace([np.inf, -np.inf], np.nan)
        out["vol_garman_klass_24"] = (
            0.5 * log_hl.pow(2) - (2.0 * np.log(2.0) - 1.0) * log_co.pow(2)
        ).rolling(24, min_periods=12).mean().pow(0.5)

        log_o = np.log(df["open"] / df["open"].shift(1)).replace([np.inf, -np.inf], np.nan)
        log_c = np.log(df["close"] / df["open"]).replace([np.inf, -np.inf], np.nan)
        yz = (
            log_o.pow(2)
            + 0.5 * log_hl.pow(2)
            - (2.0 * np.log(2.0) - 1.0) * log_c.pow(2)
        ).rolling(24, min_periods=12).mean().pow(0.5)
        out["vol_yang_zhang_24"] = yz

        for w in (48,):
            out[f"vol_realized_skew_{w}"] = ret.rolling(w, min_periods=w // 2).skew()
            out[f"vol_realized_kurt_{w}"] = ret.rolling(w, min_periods=w // 2).kurt()
            neg = ret.where(ret < 0.0)
            pos = ret.where(ret > 0.0)
            down = neg.rolling(w, min_periods=w // 2).std()
            up = pos.rolling(w, min_periods=w // 2).std()
            out[f"vol_downside_{w}"] = down.fillna(0.0)
            out[f"vol_upside_{w}"] = up.fillna(0.0)
            ratio = down / up.replace(0.0, np.nan)
            out["vol_semivariance_ratio"] = ratio.fillna(0.0).mask(up.eq(0.0), 0.0)

        rng_pct = (df["high"] - df["low"]) / close.replace(0.0, np.nan)
        rng_ma = rng_pct.rolling(24, min_periods=12).mean()
        out["vol_range_expansion_24"] = rng_pct / rng_ma.replace(0.0, np.nan)

        hi48 = close.rolling(48, min_periods=24).max()
        lo48 = close.rolling(48, min_periods=24).min()
        out["vol_breakout_pressure_48"] = np.maximum(
            (close - hi48.shift(1)) / close,
            (lo48.shift(1) - close) / close,
        )

        spike = (out["vol_ratio"] > 2.0).astype(float)
        out["vol_time_since_vol_spike"] = spike.groupby(
            (spike != spike.shift()).cumsum()
        ).cumcount()
        out["vol_regime_persistence_48"] = (out["vol_ratio"] > 1.0).rolling(
            48, min_periods=12
        ).mean()

        for lag in (1, 6):
            out[f"vol_autocorr_{lag}_48"] = ret.rolling(48, min_periods=24).apply(
                lambda x, lag=lag: pd.Series(x).autocorr(lag=lag) if len(x) > lag else np.nan,
                raw=False,
            )

        return out

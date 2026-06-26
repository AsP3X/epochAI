"""Rug-pull and manipulation proxy features from OHLCV and derivatives."""

from __future__ import annotations

import numpy as np
import pandas as pd

from epoch_ai.features.base import FeatureGroup


class ManipulationFeatures(FeatureGroup):
    """Wash-trading, wick-cluster, and positioning-divergence proxies."""

    name = "manip"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        close = df["close"]
        ret = close.pct_change()
        vol = df["volume"]
        rng = (df["high"] - df["low"]).replace(0.0, np.nan)

        vol_ma = vol.rolling(48, min_periods=12).mean()
        vol_std = vol.rolling(48, min_periods=12).std().replace(0.0, np.nan)
        vol_z = (vol - vol_ma) / vol_std
        out["manip_vol_price_div"] = vol_z.abs() * (1.0 - ret.abs().clip(0.0, 0.05) / 0.05)

        upper = (df["high"] - df[["open", "close"]].max(axis=1)) / rng
        lower = (df[["open", "close"]].min(axis=1) - df["low"]) / rng
        wick = (upper + lower).fillna(0.0)
        out["manip_wick_cluster"] = wick.rolling(24, min_periods=6).mean()

        illiq = (ret.abs() / vol.replace(0.0, np.nan)).rolling(24, min_periods=6).mean()
        illiq_mean = illiq.rolling(96, min_periods=16).mean()
        illiq_std = illiq.rolling(96, min_periods=16).std().replace(0.0, np.nan)
        out["manip_illiq_spike"] = ((illiq - illiq_mean) / illiq_std).fillna(0.0).clip(0.0, 5.0)

        out["manip_return_skew"] = ret.rolling(48, min_periods=12).skew().fillna(0.0)
        out["manip_return_kurt"] = ret.rolling(48, min_periods=12).kurt().fillna(0.0)

        gap = (df["open"] - close.shift(1)).abs() / close.shift(1).replace(0.0, np.nan)
        recovery = (close - df["open"]).abs() / gap.replace(0.0, np.nan)
        out["manip_gap_recovery"] = recovery.fillna(0.0).clip(0.0, 1.0)

        if "open_interest" in df.columns:
            oi = df["open_interest"]
            div = (np.sign(oi.diff()) != np.sign(close.diff())).astype(float)
            out["manip_oi_price_div"] = div.rolling(12, min_periods=4).mean()

        if "funding_rate" in df.columns:
            funding = df["funding_rate"]
            f_mean = funding.rolling(96, min_periods=16).mean()
            f_std = funding.rolling(96, min_periods=16).std().replace(0.0, np.nan)
            z = (funding - f_mean) / f_std.where(f_std > 0)
            out["manip_funding_extreme"] = z.abs().fillna(0.0).mask(f_std.eq(0.0), 0.0)

        if "liquidations" in df.columns:
            liq = df["liquidations"]
            baseline = liq.rolling(96, min_periods=16).mean()
            spike = liq.div(baseline.where(baseline > 0)).fillna(0.0).clip(0, 50)
            out["manip_liq_spike"] = spike

        return out

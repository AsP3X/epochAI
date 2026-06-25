"""Volatility and market-regime features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from epoch_ai.features.base import FeatureGroup


class VolatilityFeatures(FeatureGroup):
    """Realised volatility, vol-of-vol and simple regime descriptors."""

    name = "vol"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        ret = df["close"].pct_change()

        for w in (12, 24, 48, 96):
            out[f"vol_std_{w}"] = ret.rolling(w, min_periods=w // 2).std()

        # Vol-of-vol and short/long vol ratio (regime expansion/contraction).
        vol_short = ret.rolling(24, min_periods=12).std()
        vol_long = ret.rolling(96, min_periods=32).std()
        out["vol_ratio"] = vol_short / vol_long.replace(0.0, np.nan)
        out["vol_of_vol"] = vol_short.rolling(48, min_periods=12).std()

        # Parkinson high-low volatility estimator.
        hl = np.log(df["high"] / df["low"]).replace([np.inf, -np.inf], np.nan)
        out["vol_parkinson"] = (hl.pow(2) / (4.0 * np.log(2.0))).rolling(
            24, min_periods=12
        ).mean().pow(0.5)

        # Trend strength: rolling mean return divided by rolling vol (z-like).
        out["vol_trend_strength"] = ret.rolling(48, min_periods=24).mean() / vol_long.replace(
            0.0, np.nan
        )

        # Drawdown from a rolling peak (regime stress).
        roll_max = df["close"].rolling(96, min_periods=24).max()
        out["vol_drawdown"] = df["close"] / roll_max - 1.0
        return out

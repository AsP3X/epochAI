"""Microstructure / order-flow proxy features.

True order-book depth requires a live L2 feed; until that is wired in, these features
approximate microstructure dynamics from OHLCV (candle shape, volume pressure,
intrabar range) which are strong, always-available proxies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from epoch_ai.features.base import FeatureGroup


class MicrostructureFeatures(FeatureGroup):
    """Candle-shape and volume-pressure features."""

    name = "micro"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        rng = (df["high"] - df["low"]).replace(0.0, np.nan)

        out["micro_body"] = (df["close"] - df["open"]) / rng
        out["micro_upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / rng
        out["micro_lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / rng
        out["micro_range_pct"] = rng / df["close"]
        out["micro_close_loc"] = (df["close"] - df["low"]) / rng

        # Volume pressure: current vs rolling baseline and signed by direction.
        vol = df["volume"]
        vol_ma = vol.rolling(48, min_periods=12).mean()
        out["micro_vol_z"] = (vol - vol_ma) / vol.rolling(48, min_periods=12).std().replace(
            0.0, np.nan
        )
        direction = np.sign(df["close"] - df["open"])
        out["micro_signed_vol"] = (direction * vol / vol_ma.replace(0.0, np.nan)).clip(-10, 10)

        # Amihud-style illiquidity: |return| per unit volume.
        ret = df["close"].pct_change()
        out["micro_illiq"] = (ret.abs() / vol.replace(0.0, np.nan)).rolling(
            24, min_periods=6
        ).mean()
        return out

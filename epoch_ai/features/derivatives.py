"""Derivatives-specific features (funding rate, open interest, liquidations, basis).

These are among the most predictive signals for perpetual-futures markets. Each
feature degrades gracefully: if a source column is absent it is treated as zero so
the group never breaks the pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from epoch_ai.features.base import FeatureGroup


class DerivativesFeatures(FeatureGroup):
    """Funding, open-interest and liquidation dynamics."""

    name = "deriv"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)

        if "funding_rate" in df.columns:
            funding = df["funding_rate"]
            out["deriv_funding"] = funding
            out["deriv_funding_ma"] = funding.rolling(48, min_periods=8).mean()
            out["deriv_funding_chg"] = funding.diff()
            # Causal rolling z-score. When funding is *constant* over the window the
            # rolling std is 0 (deviation undefined); treat that as a neutral 0 rather
            # than NaN so a flat funding series does not blank out the whole feature
            # matrix via the pipeline's dropna. Genuine warm-up rows (std == NaN under
            # min_periods) remain NaN and are trimmed as normal warm-up.
            funding_mean = funding.rolling(96, min_periods=16).mean()
            funding_std = funding.rolling(96, min_periods=16).std()
            z = (funding - funding_mean) / funding_std.where(funding_std > 0)
            out["deriv_funding_z"] = z.mask(funding_std.eq(0.0), 0.0)

        if "open_interest" in df.columns:
            oi = df["open_interest"]
            out["deriv_oi_chg"] = oi.pct_change()
            out["deriv_oi_chg_6"] = oi.pct_change(6)
            oi_ma = oi.rolling(96, min_periods=16).mean()
            out["deriv_oi_dist"] = oi / oi_ma - 1.0
            # Rising OI with rising price = trend conviction.
            out["deriv_oi_price_div"] = np.sign(oi.diff()) * np.sign(df["close"].diff())

        if "liquidations" in df.columns:
            liq = df["liquidations"]
            out["deriv_liq"] = liq / df["close"].replace(0.0, np.nan)
            # Liquidations are sparse (often zero), so a zero rolling baseline means
            # "no spike" rather than "missing" - map it to 0 to preserve history.
            liq_baseline = liq.rolling(96, min_periods=16).mean()
            spike = liq.div(liq_baseline.where(liq_baseline > 0))
            out["deriv_liq_spike"] = spike.fillna(0.0).clip(0, 50)

        if "spot_close" in df.columns:
            spot = df["spot_close"].replace(0.0, np.nan)
            basis = (df["close"] - spot) / spot
            out["deriv_basis"] = basis
            out["deriv_basis_ma"] = basis.rolling(48, min_periods=8).mean()
            basis_mean = basis.rolling(96, min_periods=16).mean()
            basis_std = basis.rolling(96, min_periods=16).std()
            z = (basis - basis_mean) / basis_std.where(basis_std > 0)
            out["deriv_basis_z"] = z.mask(basis_std.eq(0.0), 0.0)

        return out

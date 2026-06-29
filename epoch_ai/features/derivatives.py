"""Derivatives-specific features (funding rate, open interest, liquidations, basis).

These are among the most predictive signals for perpetual-futures markets. Each
feature degrades gracefully: if a source column is absent it is treated as zero so
the group never breaks the pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from epoch_ai.features._stats import rolling_z
from epoch_ai.features.base import FeatureGroup


class DerivativesFeatures(FeatureGroup):
    """Funding, open-interest and liquidation dynamics."""

    name = "deriv"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        close = df["close"]
        ret = close.pct_change(fill_method=None)
        vol_std = ret.rolling(48, min_periods=12).std()
        funding_z: pd.Series | None = None
        oi_dist: pd.Series | None = None
        basis: pd.Series | None = None

        if "funding_rate" in df.columns:
            funding = df["funding_rate"]
            out["deriv_funding"] = funding
            out["deriv_funding_ma"] = funding.rolling(48, min_periods=8).mean()
            out["deriv_funding_chg"] = funding.diff()
            funding_mean = funding.rolling(96, min_periods=16).mean()
            funding_std = funding.rolling(96, min_periods=16).std()
            z = (funding - funding_mean) / funding_std.where(funding_std > 0)
            funding_z = z.mask(funding_std.eq(0.0), 0.0)
            out["deriv_funding_z"] = funding_z
            out["deriv_funding_cum_48"] = funding.rolling(48, min_periods=8).sum()
            out["deriv_funding_cum_96"] = funding.rolling(96, min_periods=16).sum()
            out["deriv_funding_sign_persist"] = np.sign(funding).rolling(
                24, min_periods=6
            ).mean()

        if "open_interest" in df.columns:
            oi = df["open_interest"]
            oi_chg = oi.pct_change(fill_method=None)
            out["deriv_oi_chg"] = oi_chg
            out["deriv_oi_chg_6"] = oi.pct_change(6, fill_method=None)
            oi_ma = oi.rolling(96, min_periods=16).mean()
            oi_dist = oi / oi_ma - 1.0
            out["deriv_oi_dist"] = oi_dist
            out["deriv_oi_price_div"] = np.sign(oi.diff()) * np.sign(close.diff())
            out["deriv_oi_accel"] = oi_chg.diff()
            out["deriv_oi_vol_ratio"] = oi_chg / vol_std.replace(0.0, np.nan)

        if "liquidations" in df.columns:
            liq = df["liquidations"]
            out["deriv_liq"] = liq / close.replace(0.0, np.nan)
            liq_baseline = liq.rolling(96, min_periods=16).mean()
            spike = liq.div(liq_baseline.where(liq_baseline > 0))
            out["deriv_liq_spike"] = spike.fillna(0.0).clip(0, 50)
            vol_ratio = ret.rolling(24, min_periods=12).std() / ret.rolling(
                96, min_periods=32
            ).std().replace(0.0, np.nan)
            out["deriv_liq_cascade_score"] = out["deriv_liq_spike"] * vol_ratio.fillna(0.0)

        if "spot_close" in df.columns:
            spot = df["spot_close"].replace(0.0, np.nan)
            basis = (close - spot) / spot
            out["deriv_basis"] = basis
            basis_ma = basis.rolling(48, min_periods=8).mean()
            out["deriv_basis_ma"] = basis_ma
            out["deriv_basis_term_structure"] = basis - basis_ma
            out["deriv_basis_slope_48"] = basis_ma.diff(48)
            basis_mean = basis.rolling(96, min_periods=16).mean()
            basis_std = basis.rolling(96, min_periods=16).std()
            z = (basis - basis_mean) / basis_std.where(basis_std > 0)
            out["deriv_basis_z"] = z.mask(basis_std.eq(0.0), 0.0)

        if "mark_price" in df.columns and "index_price" in df.columns:
            idx = df["index_price"].replace(0.0, np.nan)
            premium = (df["mark_price"] - idx) / idx
            out["deriv_mark_premium_pct"] = premium
            out["deriv_mark_premium_z"] = rolling_z(premium)
            out["deriv_index_basis_pct"] = premium
            out["deriv_index_basis_z"] = rolling_z(premium)

        if "premium_index" in df.columns:
            prem = df["premium_index"]
            out["deriv_premium"] = prem
            out["deriv_premium_ma"] = prem.rolling(48, min_periods=8).mean()
            out["deriv_premium_z"] = rolling_z(prem)

        if "long_short_ratio" in df.columns:
            ls = df["long_short_ratio"]
            out["deriv_ls_ratio"] = ls
            out["deriv_ls_ratio_z"] = rolling_z(ls)
            out["deriv_ls_ratio_chg"] = ls.diff()

        if "top_trader_long_short_ratio" in df.columns:
            top = df["top_trader_long_short_ratio"]
            out["deriv_top_ls_ratio"] = top
            out["deriv_top_ls_z"] = rolling_z(top)

        if "liquidations_long" in df.columns and "liquidations_short" in df.columns:
            long_liq = df["liquidations_long"]
            short_liq = df["liquidations_short"]
            out["deriv_liq_long"] = long_liq / close.replace(0.0, np.nan)
            out["deriv_liq_short"] = short_liq / close.replace(0.0, np.nan)
            imb = (long_liq - short_liq) / (long_liq + short_liq).replace(0.0, np.nan)
            out["deriv_liq_imbalance"] = imb.fillna(0.0)
            out["deriv_liq_imbalance_z"] = rolling_z(out["deriv_liq_imbalance"])

        if funding_z is not None and oi_dist is not None:
            out["deriv_oi_funding_combo"] = oi_dist * funding_z
            out["deriv_positioning_stress"] = funding_z * oi_dist

        return out

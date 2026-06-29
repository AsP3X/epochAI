"""Microstructure / order-flow proxy features.

True order-book depth requires a live L2 feed; until that is wired in, these features
approximate microstructure dynamics from OHLCV (candle shape, volume pressure,
intrabar range) which are strong, always-available proxies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from epoch_ai.features._stats import rolling_z, signed_streak
from epoch_ai.features.base import FeatureGroup


class MicrostructureFeatures(FeatureGroup):
    """Candle-shape and volume-pressure features."""

    name = "micro"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        rng = (df["high"] - df["low"]).replace(0.0, np.nan)
        close = df["close"]
        open_ = df["open"]
        body = (close - open_).abs()

        out["micro_body"] = (close - open_) / rng
        out["micro_upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / rng
        out["micro_lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / rng
        out["micro_range_pct"] = rng / close
        out["micro_close_loc"] = (close - df["low"]) / rng

        vol = df["volume"]
        vol_ma = vol.rolling(48, min_periods=12).mean()
        vol_std = vol.rolling(48, min_periods=12).std().replace(0.0, np.nan)
        out["micro_vol_z"] = (vol - vol_ma) / vol_std
        direction = np.sign(close - open_)
        out["micro_signed_vol"] = (direction * vol / vol_ma.replace(0.0, np.nan)).clip(-10, 10)

        ret = close.pct_change(fill_method=None)
        out["micro_illiq"] = (ret.abs() / vol.replace(0.0, np.nan)).rolling(
            24, min_periods=6
        ).mean()

        out["micro_streak_length"] = signed_streak(close)
        prev_close = close.shift(1)
        out["micro_gap_pct"] = (open_ - prev_close).abs() / prev_close.replace(0.0, np.nan)
        prev_high = df["high"].shift(1)
        prev_low = df["low"].shift(1)
        out["micro_inside_bar"] = (
            (df["high"] < prev_high) & (df["low"] > prev_low)
        ).astype(float)
        out["micro_outside_bar"] = (
            (df["high"] > prev_high) & (df["low"] < prev_low)
        ).astype(float)

        range_n = rng / close
        out["micro_nr4"] = (range_n == range_n.rolling(4, min_periods=4).min()).astype(float)
        out["micro_nr7"] = (range_n == range_n.rolling(7, min_periods=7).min()).astype(float)
        out["micro_climax_vol"] = out["micro_vol_z"].abs() * out["micro_range_pct"]
        out["micro_absorption"] = out["micro_vol_z"].abs() * (
            1.0 - body / rng.replace(0.0, np.nan)
        ).fillna(0.0)
        out["micro_wick_rejection_up"] = (out["micro_upper_wick"] > 2.0 * body / rng).astype(
            float
        )
        out["micro_wick_rejection_down"] = (out["micro_lower_wick"] > 2.0 * body / rng).astype(
            float
        )
        out["micro_vol_clump_24"] = out["micro_vol_z"].rolling(24, min_periods=6).std()

        if "taker_buy_volume" in df.columns and "taker_sell_volume" in df.columns:
            buy = df["taker_buy_volume"]
            sell = df["taker_sell_volume"]
            total = (buy + sell).replace(0.0, np.nan)
            imb = (buy - sell) / total
            out["micro_cvd"] = (buy - sell).cumsum()
            out["micro_cvd_slope_12"] = out["micro_cvd"].diff(12)
            out["micro_cvd_slope_48"] = out["micro_cvd"].diff(48)
            out["micro_taker_imbalance"] = imb
            out["micro_taker_imb_z"] = rolling_z(imb)

        if "bid_depth_1pct" in df.columns and "ask_depth_1pct" in df.columns:
            bid = df["bid_depth_1pct"]
            ask = df["ask_depth_1pct"]
            depth = (bid + ask).replace(0.0, np.nan)
            out["micro_book_imbalance"] = (bid - ask) / depth
            out["micro_book_imb_z"] = rolling_z(out["micro_book_imbalance"])
            out["micro_spread_bps"] = ask / bid.replace(0.0, np.nan) - 1.0
            out["micro_spread_z"] = rolling_z(out["micro_spread_bps"])

        if "trade_count" in df.columns:
            tc = df["trade_count"]
            out["micro_trade_intensity"] = tc / tc.rolling(48, min_periods=12).mean()
            out["micro_avg_trade_size"] = vol / tc.replace(0.0, np.nan)

        return out

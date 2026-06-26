"""PatternFeatures group: classic chart geometry as continuous scores."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from epoch_ai.features.base import FeatureGroup
from epoch_ai.features.patterns.geometry import (
    breakout_strength,
    candlestick_context_score,
    double_top_bottom_score,
    flag_pole_score,
    head_shoulders_score,
    triangle_convergence_score,
)
from epoch_ai.features.patterns.swings import confirmed_swing_highs, confirmed_swing_lows


class PatternFeatures(FeatureGroup):
    """Classic chart-pattern geometry (secondary signal; causal pivots)."""

    name = "pat"

    def __init__(
        self,
        lookbacks: Sequence[int] = (48, 96, 192),
        pivot_confirm_bars: int = 3,
    ) -> None:
        self.lookbacks = tuple(lookbacks)
        self.pivot_confirm_bars = pivot_confirm_bars

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        out["pat_swing_high_dist"] = confirmed_swing_highs(
            df["high"], self.pivot_confirm_bars
        )
        out["pat_swing_low_dist"] = confirmed_swing_lows(df["low"], self.pivot_confirm_bars)
        for w in self.lookbacks:
            out[f"pat_dtop_{w}"] = double_top_bottom_score(
                df, w, mode="top", pivot_confirm_bars=self.pivot_confirm_bars
            )
            out[f"pat_dbottom_{w}"] = double_top_bottom_score(
                df, w, mode="bottom", pivot_confirm_bars=self.pivot_confirm_bars
            )
            out[f"pat_tri_conv_{w}"] = triangle_convergence_score(df, w)
            out[f"pat_flag_{w}"] = flag_pole_score(df, w)
            out[f"pat_breakout_{w}"] = breakout_strength(df, w)
        hs = head_shoulders_score(df, max(self.lookbacks), self.pivot_confirm_bars)
        out["pat_hs_top"] = hs["top"]
        out["pat_hs_inv"] = hs["inv"]
        ctx = candlestick_context_score(df)
        out["pat_engulf"] = ctx["engulf"]
        out["pat_doji_ext"] = ctx["doji_ext"]
        return out

"""Cross-asset context features (e.g. ETH vs BTC).

Expects columns joined by :mod:`epoch_ai.data.enrichment` — ``eth_close``,
``eth_volume``, etc. — for each symbol in ``data.context_symbols``. All features
are causal (lags, rolling windows on past-or-current data only).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from epoch_ai.data.symbols import asset_prefix
from epoch_ai.features.base import FeatureGroup
from epoch_ai.utils.logging import get_logger

logger = get_logger(__name__)


class CrossAssetFeatures(FeatureGroup):
    """Relative strength, ratio dynamics, and rolling correlation vs context assets."""

    name = "xasset"

    def __init__(
        self,
        context_symbols: Sequence[str],
        return_lags: Sequence[int] = (1, 3, 12, 24, 48),
        corr_window: int = 48,
    ) -> None:
        self.context_symbols = tuple(context_symbols)
        self.return_lags = tuple(return_lags)
        self.corr_window = corr_window

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        emitted = False
        btc_close = df["close"]
        btc_ret = btc_close.pct_change()

        for sym in self.context_symbols:
            pfx = asset_prefix(sym)
            close_col = f"{pfx}_close"
            if close_col not in df.columns:
                continue

            ctx_close = df[close_col]
            ctx_ret = ctx_close.pct_change()
            emitted = True

            for lag in self.return_lags:
                out[f"xasset_{pfx}_ret_{lag}"] = ctx_close.pct_change(lag)

            ratio = btc_close / ctx_close.replace(0.0, np.nan)
            out[f"xasset_{pfx}_ratio"] = ratio
            out[f"xasset_{pfx}_ratio_chg"] = ratio.pct_change()

            for lag in (12, 24, 48):
                out[f"xasset_{pfx}_rel_strength_{lag}"] = (
                    btc_ret.rolling(lag, min_periods=lag).sum()
                    - ctx_ret.rolling(lag, min_periods=lag).sum()
                )

            out[f"xasset_{pfx}_corr_{self.corr_window}"] = btc_ret.rolling(
                self.corr_window, min_periods=self.corr_window // 2
            ).corr(ctx_ret)

            vol_col = f"{pfx}_volume"
            if vol_col in df.columns:
                vol = df[vol_col]
                vol_ma = vol.rolling(48, min_periods=12).mean()
                out[f"xasset_{pfx}_vol_z"] = (vol - vol_ma) / vol.rolling(
                    48, min_periods=12
                ).std().replace(0.0, np.nan)

        if not emitted:
            logger.info(
                "CrossAssetFeatures found no context columns (e.g. 'eth_close'); "
                "enable data.context_symbols and re-download."
            )
        return out

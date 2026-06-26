"""On-chain feature group.

On-chain flows (exchange net-flows, active addresses, miner activity) are strong
medium-term signals for crypto. Token-safety columns (``liquidity_usd``,
``holder_top10_pct``, ``lp_locked_pct``) support rug-risk proxies when trading alts.

Like :class:`DerivativesFeatures`, this group degrades gracefully: each feature is
emitted only when its source column is present in the frame, so enabling the group
never breaks the pipeline when on-chain data is absent.

All transforms are causal (rolling/lagged, past-or-current bar only). Wire a real data
source by joining the columns below onto the OHLCV frame before feature computation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from epoch_ai.features.base import FeatureGroup
from epoch_ai.utils.logging import get_logger

logger = get_logger(__name__)


class OnChainFeatures(FeatureGroup):
    """Exchange-flow and network-activity dynamics (when source columns exist)."""

    name = "oc"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        emitted = False

        if "exchange_netflow" in df.columns:
            flow = df["exchange_netflow"]
            out["oc_netflow"] = flow
            out["oc_netflow_z"] = (
                flow - flow.rolling(96, min_periods=16).mean()
            ) / flow.rolling(96, min_periods=16).std().replace(0.0, np.nan)
            emitted = True

        if "active_addresses" in df.columns:
            active = df["active_addresses"]
            out["oc_active_chg"] = active.pct_change()
            active_ma = active.rolling(96, min_periods=16).mean()
            out["oc_active_dist"] = active / active_ma.replace(0.0, np.nan) - 1.0
            emitted = True

        if "liquidity_usd" in df.columns:
            liq = df["liquidity_usd"]
            liq_chg = liq.pct_change()
            out["oc_liq_chg"] = liq_chg
            liq_chg_mean = liq_chg.rolling(96, min_periods=16).mean()
            liq_chg_std = liq_chg.rolling(96, min_periods=16).std().replace(0.0, np.nan)
            out["oc_liq_chg_z"] = (liq_chg - liq_chg_mean) / liq_chg_std
            emitted = True

        if "holder_top10_pct" in df.columns:
            holders = df["holder_top10_pct"]
            out["oc_holder_conc"] = holders
            out["oc_holder_conc_chg"] = holders.diff()
            emitted = True

        if "lp_locked_pct" in df.columns:
            locked = df["lp_locked_pct"]
            out["oc_lp_lock"] = locked
            out["oc_lp_unlock_velocity"] = (-locked.diff()).clip(lower=0.0)
            emitted = True

        if not emitted:
            logger.info(
                "OnChainFeatures found no on-chain columns (e.g. 'exchange_netflow', "
                "'active_addresses', 'liquidity_usd'); returning no columns."
            )
        return out

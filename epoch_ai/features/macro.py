"""Macro and cross-market context features.

Consumes daily-or-slower columns joined onto the bar grid (``btc_dominance``,
``dxy``, ``vix``, etc.). Emits only when source columns exist; enrichment or
``market_extensions`` supplies proxies offline.
"""

from __future__ import annotations

import pandas as pd

from epoch_ai.features._stats import pct_change_safe, rolling_z
from epoch_ai.features.base import FeatureGroup


class MacroFeatures(FeatureGroup):
    """BTC dominance, stablecoin supply, trad-fi risk proxies."""

    name = "macro"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)

        if "btc_dominance" in df.columns:
            dom = df["btc_dominance"]
            out["xasset_btc_dom"] = dom / 100.0
            out["xasset_btc_dom_chg"] = dom.diff()
            out["xasset_btc_dom_z"] = rolling_z(dom)

        if "total_market_cap" in df.columns:
            mcap = df["total_market_cap"]
            out["xasset_total_mcap_ret_24"] = pct_change_safe(mcap, 96)
            out["xasset_total_mcap_ret_48"] = pct_change_safe(mcap, 192)

        if "stablecoin_supply" in df.columns:
            ss = df["stablecoin_supply"]
            out["xasset_stable_supply_chg"] = pct_change_safe(ss, 96)
            out["xasset_stable_supply_z"] = rolling_z(ss.pct_change(fill_method=None))

        if "usdt_supply" in df.columns:
            usdt = df["usdt_supply"]
            out["xasset_usdt_supply_chg"] = pct_change_safe(usdt, 96)
            out["xasset_usdt_supply_z"] = rolling_z(usdt.pct_change(fill_method=None))

        if "dxy" in df.columns:
            dxy = df["dxy"]
            out["xasset_dxy_ret_24"] = pct_change_safe(dxy, 96)
            out["xasset_dxy_z"] = rolling_z(dxy)

        if "spx_ret" in df.columns:
            spx = df["spx_ret"]
            btc_ret = df["close"].pct_change(fill_method=None)
            out["xasset_spx_ret_24"] = spx.rolling(96, min_periods=24).sum()
            out["xasset_spx_corr_48"] = btc_ret.rolling(48, min_periods=24).corr(spx)

        if "gold_ret" in df.columns:
            gold = df["gold_ret"]
            btc_ret = df["close"].pct_change(fill_method=None)
            out["xasset_gold_ret_24"] = gold.rolling(96, min_periods=24).sum()
            out["xasset_gold_corr_48"] = btc_ret.rolling(48, min_periods=24).corr(gold)

        if "vix" in df.columns:
            vix = df["vix"]
            out["xasset_vix"] = vix / 100.0
            out["xasset_vix_z"] = rolling_z(vix)

        return out

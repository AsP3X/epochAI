"""Cross-asset context features (e.g. ETH/SOL vs BTC).

Expects columns joined by :mod:`epoch_ai.data.enrichment` — ``eth_close``,
``sol_funding_rate``, etc. — for each symbol in ``data.context_symbols``. Emits
price-relative signals plus context funding/OI/liquidation and OHLC micro proxies
so all capturable per-asset derivatives data is available to the model. All features
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


def _funding_z(funding: pd.Series) -> pd.Series:
    """Causal rolling z-score; flat funding → neutral 0 (matches deriv group)."""
    funding_mean = funding.rolling(96, min_periods=16).mean()
    funding_std = funding.rolling(96, min_periods=16).std()
    z = (funding - funding_mean) / funding_std.where(funding_std > 0)
    return z.mask(funding_std.eq(0.0), 0.0)


def _basis_z(series: pd.Series) -> pd.Series:
    """Causal rolling z-score for basis/spread-like series."""
    mean = series.rolling(96, min_periods=16).mean()
    std = series.rolling(96, min_periods=16).std()
    z = (series - mean) / std.where(std > 0)
    return z.mask(std.eq(0.0), 0.0)


class CrossAssetFeatures(FeatureGroup):
    """Relative strength, ratio dynamics, and context derivatives vs BTC."""

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
        btc_ret = btc_close.pct_change(fill_method=None)
        btc_funding = df.get("funding_rate")
        btc_oi = df.get("open_interest")

        for sym in self.context_symbols:
            pfx = asset_prefix(sym)
            close_col = f"{pfx}_close"
            if close_col not in df.columns:
                continue

            ctx_close = df[close_col]
            ctx_ret = ctx_close.pct_change(fill_method=None)
            emitted = True

            for lag in self.return_lags:
                out[f"xasset_{pfx}_ret_{lag}"] = ctx_close.pct_change(lag, fill_method=None)

            ratio = btc_close / ctx_close.replace(0.0, np.nan)
            out[f"xasset_{pfx}_ratio"] = ratio
            out[f"xasset_{pfx}_ratio_chg"] = ratio.pct_change(fill_method=None)

            for lag in (12, 24, 48):
                out[f"xasset_{pfx}_rel_strength_{lag}"] = (
                    btc_ret.rolling(lag, min_periods=lag).sum()
                    - ctx_ret.rolling(lag, min_periods=lag).sum()
                )

            # Human: rolling corr is undefined when the context return window is flat
            #        (zero variance, e.g. a back-filled pre-listing region). Degrade to 0
            #        like the funding/basis z-scores so those bars survive pipeline dropna
            #        instead of being discarded; warm-up NaNs are left intact.
            # Agent: CAUSAL same-window stat only; MASK zero-variance corr -> 0.
            corr = btc_ret.rolling(
                self.corr_window, min_periods=self.corr_window // 2
            ).corr(ctx_ret)
            ctx_ret_std = ctx_ret.rolling(
                self.corr_window, min_periods=self.corr_window // 2
            ).std()
            out[f"xasset_{pfx}_corr_{self.corr_window}"] = corr.mask(ctx_ret_std.eq(0.0), 0.0)

            vol_col = f"{pfx}_volume"
            if vol_col in df.columns:
                vol = df[vol_col]
                vol_ma = vol.rolling(48, min_periods=12).mean()
                out[f"xasset_{pfx}_vol_z"] = (vol - vol_ma) / vol.rolling(
                    48, min_periods=12
                ).std().replace(0.0, np.nan)

            # OHLC micro proxies on the context asset (joined open/high/low).
            open_col, high_col, low_col = f"{pfx}_open", f"{pfx}_high", f"{pfx}_low"
            if open_col in df.columns and high_col in df.columns and low_col in df.columns:
                rng = (df[high_col] - df[low_col]).replace(0.0, np.nan)
                out[f"xasset_{pfx}_range_pct"] = rng / ctx_close.replace(0.0, np.nan)
                out[f"xasset_{pfx}_close_loc"] = (ctx_close - df[low_col]) / rng
                out[f"xasset_{pfx}_body"] = (ctx_close - df[open_col]) / rng

            funding_col = f"{pfx}_funding_rate"
            if funding_col in df.columns:
                funding = df[funding_col]
                out[f"xasset_{pfx}_funding"] = funding
                out[f"xasset_{pfx}_funding_ma"] = funding.rolling(48, min_periods=8).mean()
                out[f"xasset_{pfx}_funding_chg"] = funding.diff()
                out[f"xasset_{pfx}_funding_z"] = _funding_z(funding)
                if btc_funding is not None:
                    spread = btc_funding - funding
                    out[f"xasset_{pfx}_funding_spread"] = spread
                    out[f"xasset_{pfx}_funding_spread_z"] = _basis_z(spread)

            oi_col = f"{pfx}_open_interest"
            if oi_col in df.columns:
                oi = df[oi_col]
                out[f"xasset_{pfx}_oi_chg"] = oi.pct_change(fill_method=None)
                out[f"xasset_{pfx}_oi_chg_6"] = oi.pct_change(6, fill_method=None)
                oi_ma = oi.rolling(96, min_periods=16).mean()
                out[f"xasset_{pfx}_oi_dist"] = oi / oi_ma - 1.0
                out[f"xasset_{pfx}_oi_price_div"] = np.sign(oi.diff()) * np.sign(
                    ctx_close.diff()
                )
                if btc_oi is not None:
                    btc_oi_chg = btc_oi.pct_change(fill_method=None)
                    out[f"xasset_{pfx}_oi_chg_spread"] = btc_oi_chg - oi.pct_change(
                        fill_method=None
                    )

            liq_col = f"{pfx}_liquidations"
            if liq_col in df.columns:
                liq = df[liq_col]
                out[f"xasset_{pfx}_liq"] = liq / ctx_close.replace(0.0, np.nan)
                liq_baseline = liq.rolling(96, min_periods=16).mean()
                spike = liq.div(liq_baseline.where(liq_baseline > 0))
                out[f"xasset_{pfx}_liq_spike"] = spike.fillna(0.0).clip(0, 50)

        if not emitted:
            logger.info(
                "CrossAssetFeatures found no context columns (e.g. 'eth_close'); "
                "enable data.context_symbols and re-download."
            )
            return out

        # Basket / dispersion features across all joined context assets.
        ctx_rets: list[pd.Series] = []
        fundings: list[pd.Series] = []
        oi_chgs: list[pd.Series] = []
        for sym in self.context_symbols:
            pfx = asset_prefix(sym)
            close_col = f"{pfx}_close"
            if close_col not in df.columns:
                continue
            ctx_close = df[close_col]
            ctx_ret = ctx_close.pct_change(fill_method=None)
            ctx_rets.append(ctx_ret)
            beta_cov = btc_ret.rolling(48, min_periods=24).cov(ctx_ret)
            ctx_var = ctx_ret.rolling(48, min_periods=24).var().replace(0.0, np.nan)
            out[f"xasset_{pfx}_beta_48"] = beta_cov / ctx_var
            out[f"xasset_{pfx}_lead_lag_6"] = ctx_ret.shift(6).rolling(
                48, min_periods=24
            ).corr(btc_ret)
            btc_vol = btc_ret.rolling(48, min_periods=24).std()
            ctx_vol = ctx_ret.rolling(48, min_periods=24).std()
            out[f"xasset_{pfx}_vol_ratio"] = ctx_vol / btc_vol.replace(0.0, np.nan)
            if f"{pfx}_funding_rate" in df.columns:
                fundings.append(df[f"{pfx}_funding_rate"])
            if f"{pfx}_open_interest" in df.columns:
                oi_chgs.append(df[f"{pfx}_open_interest"].pct_change(fill_method=None))

        if ctx_rets:
            basket = sum(ctx_rets) / len(ctx_rets)
            out["xasset_basket_ret_24"] = basket.rolling(24, min_periods=12).sum()
            up_frac = sum((r > 0).astype(float) for r in ctx_rets) / len(ctx_rets)
            out["xasset_alt_breadth"] = up_frac.rolling(24, min_periods=12).mean()
        if fundings and btc_funding is not None:
            all_f = pd.concat([btc_funding, *fundings], axis=1)
            out["xasset_funding_dispersion"] = all_f.std(axis=1)
        if oi_chgs and btc_oi is not None:
            all_oi = pd.concat([btc_oi.pct_change(fill_method=None), *oi_chgs], axis=1)
            out["xasset_oi_dispersion"] = all_oi.std(axis=1)

        return out

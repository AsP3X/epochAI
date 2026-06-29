"""Extend OHLCV frames with derived, macro, and on-chain columns.

Real exchange feeds may supply some columns directly; when absent this module
synthesises causal proxies from OHLCV and derivatives context so feature groups
can emit their full column set offline (synthetic fallback, tests, geo-blocked CI).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from epoch_ai.utils.logging import get_logger

logger = get_logger(__name__)


def extend_market_columns(df: pd.DataFrame, *, seed: int = 7) -> pd.DataFrame:
    """Add missing extension columns without overwriting existing ones."""
    if df.empty:
        return df
    out = df.copy()
    rng = np.random.default_rng(seed + len(out) % 997)

    out = _extend_derivatives(out, rng)
    out = _extend_microstructure_book(out, rng)
    out = _extend_macro(out, rng)
    out = _extend_onchain(out, rng)
    out = _extend_sentiment_extra(out, rng)
    added = len(set(out.columns) - set(df.columns))
    if added:
        logger.debug("Extended market frame with %d synthetic/proxy column(s).", added)
    return out


def _extend_derivatives(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    close = df["close"]
    ret = close.pct_change(fill_method=None)
    vol = df.get("volume", pd.Series(1.0, index=df.index))
    funding = df.get("funding_rate", pd.Series(0.0, index=df.index))
    liq = df.get("liquidations", pd.Series(0.0, index=df.index))

    if "mark_price" not in df.columns:
        df["mark_price"] = close * (1.0 + funding * 0.05)
    if "index_price" not in df.columns:
        spot = df.get("spot_close")
        if spot is not None:
            df["index_price"] = spot
        else:
            df["index_price"] = close * (1.0 - funding * 0.02)
    if "premium_index" not in df.columns:
        idx = df["index_price"].replace(0.0, np.nan)
        df["premium_index"] = (df["mark_price"] - idx) / idx

    if "long_short_ratio" not in df.columns:
        drift = ret.rolling(48, min_periods=12).mean()
        df["long_short_ratio"] = (1.0 + drift * 50.0).clip(0.3, 3.0)
    if "top_trader_long_short_ratio" not in df.columns:
        df["top_trader_long_short_ratio"] = df["long_short_ratio"] * (
            1.0 + 0.1 * rng.standard_normal(len(df))
        ).clip(0.5, 2.0)

    if "taker_buy_volume" not in df.columns:
        buy_frac = (0.5 + 0.25 * np.sign(ret.fillna(0.0))).clip(0.05, 0.95)
        df["taker_buy_volume"] = vol * buy_frac
    if "taker_sell_volume" not in df.columns:
        df["taker_sell_volume"] = vol - df["taker_buy_volume"]

    if "liquidations_long" not in df.columns:
        df["liquidations_long"] = liq * (ret < 0).astype(float)
    if "liquidations_short" not in df.columns:
        df["liquidations_short"] = liq * (ret > 0).astype(float)

    if "trade_count" not in df.columns:
        df["trade_count"] = (vol / close.replace(0.0, np.nan) * 1000.0).clip(lower=1.0)

    return df


def _extend_microstructure_book(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    vol = df.get("volume", pd.Series(1.0, index=df.index))
    close = df["close"]
    if "bid_depth_1pct" not in df.columns:
        df["bid_depth_1pct"] = vol * close * (0.8 + 0.2 * rng.random(len(df)))
    if "ask_depth_1pct" not in df.columns:
        df["ask_depth_1pct"] = vol * close * (0.8 + 0.2 * rng.random(len(df)))
    return df


def _extend_macro(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    close = df["close"]
    ret = close.pct_change(fill_method=None)
    n = len(df)

    if "btc_dominance" not in df.columns:
        dom = 52.0 + ret.rolling(96, min_periods=24).sum() * -200.0
        df["btc_dominance"] = dom.fillna(52.0).clip(35.0, 70.0)
    if "total_market_cap" not in df.columns:
        df["total_market_cap"] = close * 19_000_000.0
    if "stablecoin_supply" not in df.columns:
        base = 120e9
        df["stablecoin_supply"] = base * np.exp(np.cumsum(rng.standard_normal(n) * 1e-5))
    if "usdt_supply" not in df.columns:
        df["usdt_supply"] = df["stablecoin_supply"] * 0.65
    if "dxy" not in df.columns:
        df["dxy"] = 104.0 + np.cumsum(rng.standard_normal(n) * 0.02)
    if "spx_ret" not in df.columns:
        df["spx_ret"] = rng.standard_normal(n) * 0.001 + ret * 0.3
    if "gold_ret" not in df.columns:
        df["gold_ret"] = rng.standard_normal(n) * 0.0005 - ret * 0.1
    if "vix" not in df.columns:
        vol = ret.rolling(48, min_periods=12).std().fillna(0.01)
        df["vix"] = (15.0 + vol * 500.0).clip(10.0, 80.0)
    return df


def _extend_onchain(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    ret = df["close"].pct_change(fill_method=None)
    n = len(df)
    noise = rng.standard_normal(n)

    if "exchange_netflow" not in df.columns:
        df["exchange_netflow"] = -ret * 1000.0 + noise * 50.0
    if "active_addresses" not in df.columns:
        base = 800_000.0
        df["active_addresses"] = base * (1.0 + np.cumsum(noise * 0.001))
    if "exchange_reserve" not in df.columns:
        df["exchange_reserve"] = 2e6 * np.exp(np.cumsum(noise * 1e-4))
    if "miner_outflow" not in df.columns:
        df["miner_outflow"] = np.maximum(0.0, noise * 100.0)
    if "whale_transactions" not in df.columns:
        df["whale_transactions"] = (10.0 + np.abs(noise) * 5.0).clip(0.0, 100.0)
    if "mvrv" not in df.columns:
        df["mvrv"] = (1.8 + ret.rolling(96, min_periods=24).sum() * 2.0).fillna(1.8)
    if "sopr" not in df.columns:
        df["sopr"] = (1.0 + ret.rolling(24, min_periods=8).mean() * 5.0).fillna(1.0)
    if "nupl" not in df.columns:
        df["nupl"] = ret.rolling(96, min_periods=24).sum().fillna(0.0).clip(-1.0, 1.0)
    if "hash_rate" not in df.columns:
        df["hash_rate"] = 400e18 * (1.0 + np.cumsum(noise * 1e-4))
    if "liquidity_usd" not in df.columns:
        df["liquidity_usd"] = df["close"] * df.get("volume", 1.0) * 100.0
    if "holder_top10_pct" not in df.columns:
        df["holder_top10_pct"] = 0.45 + noise * 0.01
    if "lp_locked_pct" not in df.columns:
        df["lp_locked_pct"] = np.clip(0.85 + np.cumsum(noise * 0.0001), 0.0, 1.0)
    return df


def _extend_sentiment_extra(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(df)
    if "social_volume" not in df.columns:
        vol = df.get("volume", pd.Series(1.0, index=df.index))
        df["social_volume"] = vol * (1.0 + rng.random(n))
    if "google_trends_bitcoin" not in df.columns:
        df["google_trends_bitcoin"] = (50.0 + rng.standard_normal(n) * 10.0).clip(0.0, 100.0)
    if "funding_weighted_avg" not in df.columns:
        fr = df.get("funding_rate", pd.Series(0.0, index=df.index))
        df["funding_weighted_avg"] = fr.rolling(48, min_periods=8).mean()
    if "news_sentiment_score" not in df.columns:
        ret = df["close"].pct_change(fill_method=None)
        df["news_sentiment_score"] = ret.rolling(24, min_periods=8).mean().fillna(0.0) * 10.0
    return df

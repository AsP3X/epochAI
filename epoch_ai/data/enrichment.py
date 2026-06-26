"""Join external and cross-asset context onto the primary OHLCV frame.

After the primary symbol is loaded, :func:`enrich_primary_market` attaches:

* Cross-asset columns from ``data.context_symbols`` (e.g. ETH close/volume/funding/OI).
* Crypto Fear & Greed index (daily, forward-filled to the bar grid).
* Spot reference close for perp basis features (when ``fetch_spot_basis`` is enabled).

All joins are causal: only past-or-same-bar values via forward-fill onto the primary
index — no backward fill from the future.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from epoch_ai.data.symbols import asset_prefix
from epoch_ai.utils.logging import get_logger
from epoch_ai.utils.timeframe import timeframe_to_minutes

if TYPE_CHECKING:
    from epoch_ai.config.settings import AppConfig
    from epoch_ai.data.downloader import HistoricalDownloader

logger = get_logger(__name__)

_FNG_URL = "https://api.alternative.me/fng/?limit=0&format=json"
_FNG_CACHE = "fear_greed.parquet"
# Human: Mirror primary-market columns so context assets feed the same feature families.
# Agent: JOIN causal ffill; prefixes via asset_prefix (eth_close, sol_funding_rate, …).
_CONTEXT_JOIN_COLS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "funding_rate",
    "open_interest",
    "liquidations",
)


def enrich_primary_market(
    df: pd.DataFrame,
    config: AppConfig,
    downloader: HistoricalDownloader,
) -> pd.DataFrame:
    """Attach configured context columns to the primary market frame."""
    if df.empty:
        return df

    out = df.copy()
    data_cfg = config.data

    for ctx_symbol in data_cfg.context_symbols:
        if ctx_symbol == config.primary_symbol:
            continue
        out = _join_context_symbol(out, ctx_symbol, config, downloader)

    if data_cfg.fetch_fear_greed:
        out = _join_fear_greed(out, Path(data_cfg.data_dir))

    if data_cfg.fetch_spot_basis and data_cfg.market_type == "future":
        out = _join_spot_reference(out, config, downloader)

    return out


def _join_context_symbol(
    df: pd.DataFrame,
    ctx_symbol: str,
    config: AppConfig,
    downloader: HistoricalDownloader,
) -> pd.DataFrame:
    """Align a context symbol's OHLCV(+derivatives) onto the primary index."""
    pfx = asset_prefix(ctx_symbol)
    try:
        ctx = downloader.load_or_download(ctx_symbol, skip_enrichment=True)
    except Exception as exc:  # noqa: BLE001 - context is best-effort
        logger.warning("Context symbol %s unavailable: %s", ctx_symbol, exc)
        return df

    if ctx.empty:
        logger.warning("Context symbol %s returned no rows; skipping join.", ctx_symbol)
        return df

    ctx = ctx.reindex(df.index, method="ffill")
    joined = 0
    for col in _CONTEXT_JOIN_COLS:
        if col in ctx.columns:
            df[f"{pfx}_{col}"] = ctx[col]
            joined += 1

    if joined:
        logger.info(
            "Joined %d context column(s) from %s onto primary frame (%d bars).",
            joined,
            ctx_symbol,
            len(df),
        )
    else:
        logger.info("Context symbol %s had no joinable columns.", ctx_symbol)
    return df


def _join_fear_greed(df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Attach daily Fear & Greed index forward-filled to the bar index."""
    cache = data_dir / _FNG_CACHE
    series = _load_fear_greed_series(cache)
    if series is None or series.empty:
        logger.info("Fear & Greed index unavailable; sentiment features will no-op.")
        return df

    aligned = series.reindex(df.index, method="ffill")
    df["fear_greed"] = aligned
    logger.info(
        "Joined fear_greed onto %d bars (%d non-null).",
        len(df),
        int(df["fear_greed"].notna().sum()),
    )
    return df


def _load_fear_greed_series(cache: Path) -> pd.Series | None:
    """Load Fear & Greed from cache or the public Alternative.me API."""
    if cache.exists():
        try:
            cached = pd.read_parquet(cache)
            if "fear_greed" in cached.columns and len(cached) > 0:
                age_h = (datetime.now(UTC) - cached.index.max()).total_seconds() / 3600.0
                if age_h < 24:
                    return cached["fear_greed"]
        except Exception:  # noqa: BLE001 - refresh from API on corrupt cache
            pass

    try:
        with urllib.request.urlopen(_FNG_URL, timeout=30) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        logger.info("Fear & Greed API fetch failed: %s", exc)
        if cache.exists():
            cached = pd.read_parquet(cache)
            return cached.get("fear_greed")
        return None

    entries = payload.get("data", [])
    if not entries:
        return None

    rows = []
    for item in entries:
        ts = pd.to_datetime(int(item["timestamp"]), unit="s", utc=True)
        rows.append((ts, float(item["value"])))

    fng = pd.Series(
        [v for _, v in rows],
        index=pd.DatetimeIndex([t for t, _ in rows]),
        name="fear_greed",
    ).sort_index()
    fng = fng[~fng.index.duplicated(keep="last")]

    cache.parent.mkdir(parents=True, exist_ok=True)
    fng.to_frame().to_parquet(cache)
    logger.info("Cached %d Fear & Greed observations to %s", len(fng), cache)
    return fng


def _join_spot_reference(
    df: pd.DataFrame,
    config: AppConfig,
    downloader: HistoricalDownloader,
) -> pd.DataFrame:
    """Attach spot ``close`` as ``spot_close`` for perp basis features."""
    spot_symbol = config.data.spot_symbol or config.primary_symbol
    spot_exchange = config.data.spot_exchange

    cache = Path(config.data.data_dir) / f"spot_{spot_symbol.replace('/', '-')}_{config.timeframe}.parquet"

    spot = _load_spot_close(spot_symbol, spot_exchange, config, downloader, cache)
    if spot is None or spot.empty:
        logger.info("Spot reference unavailable for basis features.")
        return df

    aligned = spot.reindex(df.index, method="ffill")
    df["spot_close"] = aligned
    logger.info(
        "Joined spot_close for %s (%d non-null bars).",
        spot_symbol,
        int(df["spot_close"].notna().sum()),
    )
    return df


def _load_spot_close(
    spot_symbol: str,
    spot_exchange: str,
    config: AppConfig,
    downloader: HistoricalDownloader,
    cache: Path,
) -> pd.Series | None:
    """Download or load cached spot closes aligned to the futures history window."""
    if cache.exists():
        try:
            cached = pd.read_parquet(cache)
            if "spot_close" in cached.columns:
                return cached["spot_close"]
        except Exception:  # noqa: BLE001
            pass

    try:
        import ccxt  # noqa: PLC0415
    except ImportError:
        return _spot_from_context_downloader(spot_symbol, config, downloader, cache)

    exchange_cls = getattr(ccxt, spot_exchange, None)
    if exchange_cls is None:
        logger.info("Unknown spot exchange %s; skipping basis.", spot_exchange)
        return None

    try:
        exchange = exchange_cls({"enableRateLimit": True})
        timeframe = config.timeframe
        tf_ms = timeframe_to_minutes(timeframe) * 60_000
        # Human: reuse downloader's earliest/start logic via a throwaway futures exchange instance.
        since = downloader._start_since_ms(exchange)  # noqa: SLF001
        limit = 1000
        partial: pd.DataFrame | None = None
        target = downloader._default_bar_count()  # noqa: SLF001

        while partial is None or len(partial) < target:
            batch = exchange.fetch_ohlcv(spot_symbol, timeframe=timeframe, since=since, limit=limit)
            if not batch:
                break
            batch_df = downloader._rows_to_dataframe(batch)  # noqa: SLF001
            partial = batch_df if partial is None else pd.concat([partial, batch_df])
            partial = partial[~partial.index.duplicated(keep="last")].sort_index()
            since = int(partial.index[-1].timestamp() * 1000) + tf_ms
            if len(batch) < limit:
                break

        if partial is None or partial.empty:
            return None

        spot_close = partial["close"].rename("spot_close")
        cache.parent.mkdir(parents=True, exist_ok=True)
        spot_close.to_frame().to_parquet(cache)
        return spot_close
    except Exception as exc:  # noqa: BLE001
        logger.info("Spot OHLCV download failed for %s: %s", spot_symbol, exc)
        return _spot_from_context_downloader(spot_symbol, config, downloader, cache)


def _spot_from_context_downloader(
    spot_symbol: str,
    config: AppConfig,
    downloader: HistoricalDownloader,
    cache: Path,
) -> pd.Series | None:
    """Fallback: reuse the main downloader with spot market type override."""
    if spot_symbol != config.primary_symbol:
        return None
    try:
        spot_cfg = config.model_copy(deep=True)
        spot_cfg.data.market_type = "spot"
        spot_cfg.data.exchange = config.data.spot_exchange
        from epoch_ai.data.downloader import HistoricalDownloader as HD

        spot_df = HD(spot_cfg).load_or_download(spot_symbol, skip_enrichment=True)
        if spot_df.empty:
            return None
        spot_close = spot_df["close"].rename("spot_close")
        cache.parent.mkdir(parents=True, exist_ok=True)
        spot_close.to_frame().to_parquet(cache)
        return spot_close
    except Exception as exc:  # noqa: BLE001
        logger.info("Spot fallback download failed: %s", exc)
        return None

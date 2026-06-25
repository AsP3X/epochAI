"""Historical data downloader (CCXT) with an offline synthetic fallback.

The downloader fetches the **longest possible** OHLCV history for a symbol via CCXT,
paginating forward from ``historical_start_date``. For derivatives markets it also
attaches funding-rate history when the exchange supports it. If the exchange cannot
be reached (e.g. geo-blocking, no network) it transparently falls back to the
synthetic generator so the rest of the pipeline always has data to work with.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from epoch_ai.config.settings import AppConfig
from epoch_ai.data.cleaning import align_and_clean
from epoch_ai.data.synthetic import generate_synthetic_ohlcv
from epoch_ai.utils.logging import get_logger
from epoch_ai.utils.timeframe import timeframe_to_minutes

logger = get_logger(__name__)

_OHLCV_COLS = ["open", "high", "low", "close", "volume"]


class HistoricalDownloader:
    """Fetch and cache historical market data for a configured symbol."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.data_dir = Path(config.data.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ paths
    def _cache_path(self, symbol: str) -> Path:
        safe = symbol.replace("/", "-").replace(":", "_")
        return self.data_dir / f"{safe}_{self.config.timeframe}.parquet"

    # --------------------------------------------------------------- public API
    def load_or_download(
        self,
        symbol: str | None = None,
        *,
        n_bars: int | None = None,
        force: bool = False,
    ) -> pd.DataFrame:
        """Return cleaned data for ``symbol``, using cache when available.

        Args:
            symbol: Trading pair; defaults to the primary configured symbol.
            n_bars: Approximate number of bars to ensure (used for synthetic and as
                a lower bound for the cache). When ``None`` a multi-year default is
                derived from ``historical_start_date``.
            force: Re-download even if a cache file exists.

        Returns:
            A cleaned OHLCV(+context) DataFrame indexed by ``timestamp``.
        """
        symbol = symbol or self.config.primary_symbol
        cache = self._cache_path(symbol)
        if cache.exists() and not force:
            df = pd.read_parquet(cache)
            if n_bars is None or len(df) >= n_bars:
                logger.info("Loaded %d cached bars for %s from %s", len(df), symbol, cache)
                return df

        target_bars = n_bars or self._default_bar_count()
        df = self._download(symbol, target_bars)
        df = align_and_clean(df, self.config.timeframe)
        df.to_parquet(cache)
        logger.info("Saved %d bars for %s to %s", len(df), symbol, cache)
        return df

    # ------------------------------------------------------------- internals
    def _default_bar_count(self) -> int:
        """Estimate bars between ``historical_start_date`` and now."""
        start = datetime.fromisoformat(self.config.data.historical_start_date).replace(
            tzinfo=UTC
        )
        minutes = (datetime.now(UTC) - start).total_seconds() / 60.0
        return max(1, int(minutes / timeframe_to_minutes(self.config.timeframe)))

    def _download(self, symbol: str, target_bars: int) -> pd.DataFrame:
        """Try CCXT first; fall back to synthetic data on any failure."""
        try:
            df = self._download_ccxt(symbol, target_bars)
            if df is not None and len(df) > 0:
                return df
            logger.warning("CCXT returned no data for %s; using synthetic fallback.", symbol)
        except Exception as exc:  # noqa: BLE001 - any failure should fall back
            if not self.config.data.use_synthetic_fallback:
                raise
            logger.warning("CCXT download failed (%s); using synthetic fallback.", exc)

        if not self.config.data.use_synthetic_fallback:
            raise RuntimeError(f"No data available for {symbol} and synthetic fallback disabled.")

        return generate_synthetic_ohlcv(
            timeframe=self.config.timeframe,
            start=self.config.data.historical_start_date,
            n_bars=target_bars,
            seed=self.config.data.synthetic_seed,
        )

    def _download_ccxt(self, symbol: str, target_bars: int) -> pd.DataFrame | None:
        """Paginate OHLCV from the exchange via CCXT (if installed/reachable)."""
        try:
            import ccxt  # noqa: PLC0415 - optional dependency, imported lazily
        except ImportError:
            logger.info("ccxt not installed; skipping live download.")
            return None

        exchange_cls = getattr(ccxt, self.config.data.exchange, None)
        if exchange_cls is None:
            raise ValueError(f"Unknown ccxt exchange: {self.config.data.exchange}")

        exchange = exchange_cls({"enableRateLimit": True})
        timeframe = self.config.timeframe
        tf_ms = timeframe_to_minutes(timeframe) * 60_000
        since = exchange.parse8601(f"{self.config.data.historical_start_date}T00:00:00Z")

        rows: list[list[float]] = []
        limit = 1000
        while len(rows) < target_bars:
            batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
            if not batch:
                break
            rows.extend(batch)
            since = batch[-1][0] + tf_ms
            if len(batch) < limit:
                break

        if not rows:
            return None

        df = pd.DataFrame(rows, columns=["ts", *_OHLCV_COLS]).drop_duplicates("ts")
        df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.drop(columns="ts").set_index("timestamp").sort_index()
        df = self._attach_funding(exchange, symbol, df)
        return df

    def _attach_funding(self, exchange, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """Best-effort attach of funding-rate history for derivatives markets."""
        if self.config.data.market_type != "future":
            return df
        if not getattr(exchange, "has", {}).get("fetchFundingRateHistory"):
            return df
        try:
            since = exchange.parse8601(f"{self.config.data.historical_start_date}T00:00:00Z")
            history = exchange.fetch_funding_rate_history(symbol, since=since, limit=1000)
            if history:
                fr = pd.DataFrame(history)
                fr["timestamp"] = pd.to_datetime(fr["timestamp"], unit="ms", utc=True)
                fr = fr.set_index("timestamp")["fundingRate"].rename("funding_rate")
                df = df.join(fr, how="left")
        except Exception as exc:  # noqa: BLE001 - funding is best-effort context
            logger.info("Funding-rate history unavailable for %s: %s", symbol, exc)
        return df

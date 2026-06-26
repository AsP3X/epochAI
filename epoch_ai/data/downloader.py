"""Historical data downloader (CCXT) with an offline synthetic fallback.

The downloader fetches the **longest possible** OHLCV history for a symbol via CCXT,
paginating forward from ``historical_start_date`` (or from the exchange's very first
available candle when that is set to ``"earliest"``/``"auto"``). For derivatives
markets it also attaches funding-rate history when the exchange supports it. If the
exchange cannot
be reached (e.g. geo-blocking, no network) it transparently falls back to the
synthetic generator so the rest of the pipeline always has data to work with.

When a parquet cache exists but holds fewer bars than requested, the downloader
**extends** the cache from the last stored timestamp instead of re-fetching from
scratch.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from epoch_ai.config.settings import AppConfig
from epoch_ai.data.cleaning import align_and_clean
from epoch_ai.data.synthetic import generate_synthetic_ohlcv
from epoch_ai.utils.logging import get_logger
from epoch_ai.utils.progress import DownloadProgressBar, estimate_parquet_bytes, format_bytes
from epoch_ai.utils.timeframe import timeframe_to_minutes

logger = get_logger(__name__)

_OHLCV_COLS = ["open", "high", "low", "close", "volume"]
# Binance openInterestHist retains only the latest ~30 days regardless of startTime.
_OI_MAX_LOOKBACK_MS = 30 * 24 * 60 * 60 * 1000


class HistoricalDownloader:
    """Fetch and cache historical market data for a configured symbol."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.data_dir = Path(config.data.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------paths
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
        skip_enrichment: bool = False,
    ) -> pd.DataFrame:
        """Return cleaned data for ``symbol``, using cache when available.

        The parquet cache always stores the **full/longest** history downloaded so
        far. When ``n_bars`` is given, the caller receives the **most recent**
        ``n_bars`` rows (a tail slice), so consumers like ``run``/``backtest`` work on
        recent data even when the cache holds far more.

        Args:
            symbol: Trading pair; defaults to the primary configured symbol.
            n_bars: Number of most-recent bars to return. When ``None`` a multi-year
                default is derived from ``historical_start_date`` and the full cache
                is returned.
            force: Re-download even if a cache file exists.
            skip_enrichment: When ``True``, skip cross-asset/sentiment/basis joins
                (used when loading context symbols).

        Returns:
            A cleaned OHLCV(+context) DataFrame indexed by ``timestamp``.
        """
        symbol = symbol or self.config.primary_symbol
        cache = self._cache_path(symbol)
        cached: pd.DataFrame | None = None
        if cache.exists() and not force:
            cached = pd.read_parquet(cache)
            if n_bars is None or len(cached) >= n_bars:
                logger.info("Loaded %d cached bars for %s from %s", len(cached), symbol, cache)
                return self._finalize_load(
                    self._tail(cached, n_bars),
                    symbol,
                    skip_enrichment=skip_enrichment,
                )

        target_bars = n_bars or self._default_bar_count()
        if cached is not None and len(cached) > 0:
            logger.info(
                "Extending cached %s history: %d -> %d bars (%s)",
                symbol,
                len(cached),
                target_bars,
                cache,
            )
            df = self._download(symbol, target_bars, base_df=cached)
        else:
            df = self._download(symbol, target_bars)
        df = align_and_clean(df, self.config.timeframe)
        df.to_parquet(cache)
        logger.info("Saved %d bars for %s to %s", len(df), symbol, cache)
        return self._finalize_load(self._tail(df, n_bars), symbol, skip_enrichment=skip_enrichment)

    def _finalize_load(
        self,
        df: pd.DataFrame,
        symbol: str,
        *,
        skip_enrichment: bool,
    ) -> pd.DataFrame:
        """Optionally enrich the primary symbol with cross-asset and alt data."""
        if skip_enrichment or symbol != self.config.primary_symbol:
            return df
        from epoch_ai.data.enrichment import enrich_primary_market

        return enrich_primary_market(df, self.config, self)

    @staticmethod
    def _tail(df: pd.DataFrame, n_bars: int | None) -> pd.DataFrame:
        """Return the most recent ``n_bars`` rows (full frame when ``n_bars`` is None)."""
        if n_bars is None or len(df) <= n_bars:
            return df
        return df.iloc[-n_bars:].copy()

    # ------------------------------------------------------------- internals
    def _default_bar_count(self) -> int:
        """Estimate bars between the resolved start date and now."""
        start = datetime.fromisoformat(self.config.data.start_date_iso()).replace(tzinfo=UTC)
        minutes = (datetime.now(UTC) - start).total_seconds() / 60.0
        return max(1, int(minutes / timeframe_to_minutes(self.config.timeframe)))

    def _download(
        self,
        symbol: str,
        target_bars: int,
        *,
        base_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Try CCXT first; fall back to synthetic data on any failure."""
        base_len = len(base_df) if base_df is not None else 0
        if base_len >= target_bars:
            return base_df.iloc[:target_bars].copy()  # type: ignore[union-attr]

        ccxt_reason: str | None = None
        try:
            df = self._download_ccxt(symbol, target_bars, base_df=base_df)
            if df is not None and len(df) > 0:
                return df
            ccxt_reason = "CCXT returned no data"
        except Exception as exc:  # noqa: BLE001 - any failure should fall back
            ccxt_reason = f"CCXT download failed ({exc})"

        if not self.config.data.use_synthetic_fallback:
            detail = ccxt_reason or "CCXT unavailable"
            if base_df is not None and len(base_df) > 0:
                raise RuntimeError(
                    f"Could not extend cached data for {symbol} to {target_bars} bars "
                    f"({detail}). {len(base_df)} cached bars remain at {self._cache_path(symbol)}."
                )
            raise RuntimeError(
                f"No data available for {symbol} and synthetic fallback disabled ({detail}). "
                "Install ccxt (requirements-optional.txt), enable data.use_synthetic_fallback, "
                "or provide a cached parquet file under data.data_dir."
            )

        if base_df is not None and len(base_df) > 0:
            logger.warning(
                "%s for %s; keeping %d cached bars (target %d).",
                ccxt_reason,
                symbol,
                len(base_df),
                target_bars,
            )
            return base_df

        logger.warning("%s for %s; using synthetic fallback.", ccxt_reason, symbol)

        with DownloadProgressBar(
            total=target_bars,
            desc=f"Synthesizing {symbol}",
        ) as progress:
            df = generate_synthetic_ohlcv(
                timeframe=self.config.timeframe,
                start=self.config.data.start_date_iso(),
                n_bars=target_bars,
                seed=self._synthetic_seed(symbol),
            )
            progress.begin_rate_tracking()
            progress.advance_to(len(df))
        logger.info(
            "Synthetic data ready: %d bars (~%s)",
            len(df),
            format_bytes(estimate_parquet_bytes(len(df))),
        )
        return df

    def _synthetic_seed(self, symbol: str) -> int:
        """Deterministic but distinct seed per symbol for synthetic fallback."""
        offset = sum(ord(c) for c in symbol) % 997
        return self.config.data.synthetic_seed + offset

    def _start_since_ms(self, exchange) -> int:
        """Resolve the ``since`` epoch-ms for the first fetch.

        In ``earliest`` mode we hand the exchange a pre-crypto sentinel date so it
        clamps to its very first available candle (true full history). Otherwise we
        start from the configured ISO date.
        """
        if self.config.data.fetch_from_earliest():
            # 2010-01-01: older than any crypto OHLCV; exchange returns its earliest bar.
            return exchange.parse8601("2010-01-01T00:00:00Z")
        return exchange.parse8601(f"{self.config.data.historical_start_date}T00:00:00Z")

    @staticmethod
    def _rows_to_dataframe(rows: list[list[float]]) -> pd.DataFrame:
        df = pd.DataFrame(rows, columns=["ts", *_OHLCV_COLS]).drop_duplicates("ts")
        df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df.drop(columns="ts").set_index("timestamp").sort_index()

    def _download_ccxt(
        self,
        symbol: str,
        target_bars: int,
        *,
        base_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame | None:
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

        if base_df is not None and not base_df.empty:
            since = int(base_df.index.max().timestamp() * 1000) + tf_ms
            desc = f"Extending {symbol}"
            partial: pd.DataFrame | None = base_df.copy()
        else:
            since = self._start_since_ms(exchange)
            desc = f"Downloading {symbol}"
            partial = None

        limit = 1000
        with DownloadProgressBar(total=target_bars, desc=desc) as progress:
            start = len(partial) if partial is not None else 0
            progress.advance_to(start, render=False)
            progress.begin_rate_tracking()
            progress.refresh()

            while partial is None or len(partial) < target_bars:
                prev_len = 0 if partial is None else len(partial)
                batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
                if not batch:
                    break
                batch_df = self._rows_to_dataframe(batch)
                if partial is None:
                    partial = batch_df
                else:
                    partial = pd.concat([partial, batch_df])
                    partial = partial[~partial.index.duplicated(keep="last")].sort_index()

                if len(partial) == prev_len:
                    logger.warning(
                        "Exchange returned no new bars for %s; stopping at %d unique bars.",
                        symbol,
                        len(partial),
                    )
                    break

                progress.advance_to(len(partial))
                since = int(partial.index[-1].timestamp() * 1000) + tf_ms
                if len(batch) < limit:
                    break

            final_len = 0 if partial is None else len(partial)
            if final_len < target_bars:
                progress.set_total(max(final_len, 1))

        if partial is None or partial.empty:
            return None

        if final_len < target_bars:
            logger.info(
                "Exchange history ends at %d bars for %s (requested %d).",
                final_len,
                symbol,
                target_bars,
            )

        df = partial.iloc[:target_bars]
        df = self._attach_funding(exchange, symbol, df)
        if self.config.data.fetch_open_interest:
            df = self._attach_open_interest(exchange, symbol, df)
        return df

    def _attach_funding(self, exchange, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """Paginate funding-rate history for derivatives markets."""
        if self.config.data.market_type != "future":
            return df
        if not getattr(exchange, "has", {}).get("fetchFundingRateHistory"):
            return df
        try:
            since = self._start_since_ms(exchange)
            end_ms = int(df.index.max().timestamp() * 1000)
            limit = 1000
            chunks: list[pd.Series] = []
            while since <= end_ms:
                history = exchange.fetch_funding_rate_history(symbol, since=since, limit=limit)
                if not history:
                    break
                fr = pd.DataFrame(history)
                fr["timestamp"] = pd.to_datetime(fr["timestamp"], unit="ms", utc=True)
                series = fr.set_index("timestamp")["fundingRate"].rename("funding_rate")
                chunks.append(series)
                since = int(history[-1]["timestamp"]) + 1
                if len(history) < limit:
                    break

            if not chunks:
                return df

            funding = pd.concat(chunks).sort_index()
            funding = funding[~funding.index.duplicated(keep="last")]
            aligned = funding.reindex(df.index, method="ffill")
            if "funding_rate" in df.columns:
                df["funding_rate"] = df["funding_rate"].combine_first(aligned)
            else:
                df["funding_rate"] = aligned
            logger.info(
                "Attached paginated funding for %s (%d points, %d bars filled).",
                symbol,
                len(funding),
                int(df["funding_rate"].notna().sum()),
            )
        except Exception as exc:  # noqa: BLE001 - funding is best-effort context
            logger.info("Funding-rate history unavailable for %s: %s", symbol, exc)
        return df

    def _attach_open_interest(self, exchange, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """Paginate open-interest history when the exchange supports it."""
        if self.config.data.market_type != "future":
            return df
        if not getattr(exchange, "has", {}).get("fetchOpenInterestHistory"):
            return df
        try:
            end_ms = int(df.index.max().timestamp() * 1000)
            configured_since = self._start_since_ms(exchange)
            # Human: Binance rejects startTime outside its ~30-day OI retention window.
            # Agent: CLAMP since to max(configured_since, end_ms - 30d); CAUSAL ffill only.
            since = max(configured_since, end_ms - _OI_MAX_LOOKBACK_MS)
            timeframe = self.config.timeframe
            limit = 500
            chunks: list[pd.Series] = []
            while since <= end_ms:
                history = exchange.fetch_open_interest_history(
                    symbol, timeframe=timeframe, since=since, limit=limit
                )
                if not history:
                    break
                oi_df = pd.DataFrame(history)
                oi_df["timestamp"] = pd.to_datetime(oi_df["timestamp"], unit="ms", utc=True)
                value_col = "openInterestValue" if "openInterestValue" in oi_df.columns else "openInterestAmount"
                series = oi_df.set_index("timestamp")[value_col].rename("open_interest")
                chunks.append(series)
                since = int(history[-1]["timestamp"]) + 1
                if len(history) < limit:
                    break

            if not chunks:
                return df

            oi = pd.concat(chunks).sort_index()
            oi = oi[~oi.index.duplicated(keep="last")]
            aligned = oi.reindex(df.index, method="ffill")
            if "open_interest" in df.columns:
                df["open_interest"] = df["open_interest"].combine_first(aligned)
            else:
                df["open_interest"] = aligned
            logger.info(
                "Attached paginated open interest for %s (%d points, %d bars filled).",
                symbol,
                len(oi),
                int(df["open_interest"].notna().sum()),
            )
        except Exception as exc:  # noqa: BLE001 - OI is best-effort context
            logger.info("Open-interest history unavailable for %s: %s", symbol, exc)
        return df

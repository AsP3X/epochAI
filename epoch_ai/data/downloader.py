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
from epoch_ai.data.provenance import (
    SOURCE_EXCHANGE,
    SOURCE_SYNTHETIC,
    read_data_provenance,
    write_data_provenance,
)
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
        # Memoized earliest-available candle per symbol (probed lazily; see
        # _exchange_earliest_ts). Avoids re-probing within one downloader instance.
        self._earliest_ts_cache: dict[str, pd.Timestamp | None] = {}

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
        align_index: pd.DatetimeIndex | None = None,
        force: bool = False,
        fetch_if_missing: bool = True,
        skip_enrichment: bool = False,
        full_history: bool = False,
    ) -> pd.DataFrame:
        """Return cleaned data for ``symbol``, using cache when available.

        The parquet cache always stores the **full/longest** history downloaded so
        far. When ``n_bars`` is given, the caller receives the **most recent**
        ``n_bars`` rows (a tail slice), so consumers like ``run``/``backtest`` work on
        recent data even when the cache holds far more.

        When ``align_index`` is set (cross-asset joins), data is loaded for that
        **timestamp window** instead of the most recent ``n_bars`` tail.

        Args:
            symbol: Trading pair; defaults to the primary configured symbol.
            n_bars: Number of most-recent bars to return. When ``None`` a multi-year
                default is derived from ``historical_start_date`` and the full cache
                is returned.
            align_index: Primary bar index; fetch context covering this window.
            force: Re-download even if a cache file exists.
            fetch_if_missing: When ``False``, use parquet cache only; never call the
                exchange (``train`` default). Raises if cache is missing or too small.
            skip_enrichment: When ``True``, skip cross-asset/sentiment/basis joins
                (used when loading context symbols).
            full_history: When ``True`` with ``n_bars=None``, backfill from exchange
                start even when the cache already holds a long recent tail.

        Returns:
            A cleaned OHLCV(+context) DataFrame indexed by ``timestamp``.
        """
        symbol = symbol or self.config.primary_symbol
        if align_index is not None and len(align_index) > 0:
            return self._load_for_window(
                symbol,
                align_index.min(),
                align_index.max(),
                force=force,
                fetch_if_missing=fetch_if_missing,
                skip_enrichment=skip_enrichment,
            )

        cache = self._cache_path(symbol)
        if not fetch_if_missing and not force:
            if not cache.exists():
                raise RuntimeError(
                    f"No cached data for {symbol} under {self.data_dir}. "
                    f"Run: python -m epoch_ai download --bars {n_bars or 'N'}"
                )
            cached = pd.read_parquet(cache)
            if n_bars is not None and len(cached) < n_bars:
                raise RuntimeError(
                    f"Cache has {len(cached)} bars for {symbol} ({cache}) but "
                    f"{n_bars} requested. Run: python -m epoch_ai download --bars {n_bars}"
                )
            logger.info(
                "Loaded %d cached bars for %s from %s (cache-only)",
                min(len(cached), n_bars) if n_bars else len(cached),
                symbol,
                cache,
            )
            return self._finalize_load(
                self._tail(cached, n_bars),
                symbol,
                skip_enrichment=skip_enrichment,
                fetch_if_missing=False,
            )

        cached: pd.DataFrame | None = None
        if cache.exists() and not force:
            cached = pd.read_parquet(cache)
            if n_bars is None and not full_history:
                # A live cache that already reaches as far back as the exchange offers is
                # complete; return it without re-fetching. Do NOT gate on a raw bar-count
                # target: an ``earliest`` start over-estimates available bars (it assumes
                # data back to the fallback date), so a count gate would never be met and
                # the downloader would needlessly re-extend on every run.
                if self._cache_is_live(cached) and self._cache_covers_start(cached, symbol):
                    logger.info("Loaded %d cached bars for %s from %s", len(cached), symbol, cache)
                    return self._finalize_load(
                        self._tail(cached, n_bars),
                        symbol,
                        skip_enrichment=skip_enrichment,
                    )
            elif len(cached) >= n_bars and self._cache_is_live(cached):
                logger.info("Loaded %d cached bars for %s from %s", len(cached), symbol, cache)
                return self._finalize_load(
                    self._tail(cached, n_bars),
                    symbol,
                    skip_enrichment=skip_enrichment,
                )

        target_bars = n_bars or self._default_bar_count()
        recent_tail = n_bars is not None
        if cached is not None and len(cached) > 0 and not recent_tail:
            if len(cached) >= target_bars or self._cache_covers_start(cached, symbol):
                logger.info(
                    "Extending cached %s history: %d -> %d bars (%s)",
                    symbol,
                    len(cached),
                    target_bars,
                    cache,
                )
                df, source = self._download(symbol, target_bars, base_df=cached)
            else:
                logger.info(
                    "Backfilling %s from exchange start (cache %d bars, target %d; "
                    "may take a long time) %s",
                    symbol,
                    len(cached),
                    target_bars,
                    cache,
                )
                fetched, fetch_source = self._download(symbol, target_bars, recent_tail=False)
                df = self._merge_cached(cached, fetched)
                source = self._merged_provenance_source(cache, fetch_source)
        elif cached is not None and len(cached) > 0 and recent_tail:
            logger.info(
                "Refreshing recent %s history (%d bars requested; cache ends %s)",
                symbol,
                target_bars,
                cached.index.max(),
            )
            try:
                df, fetch_source = self._download(symbol, target_bars, recent_tail=True)
                df = self._merge_cached(cached, df)
                source = self._merged_provenance_source(cache, fetch_source)
            except RuntimeError:
                logger.warning(
                    "Recent refresh failed for %s; using cached bars ending %s",
                    symbol,
                    cached.index.max(),
                )
                df = cached
                source = self._existing_provenance_source(cache)
        else:
            df, source = self._download(symbol, target_bars, recent_tail=recent_tail)
        df = align_and_clean(df, self.config.timeframe)
        self._write_cache(df, cache, symbol, source)
        return self._finalize_load(self._tail(df, n_bars), symbol, skip_enrichment=skip_enrichment)

    def _load_for_window(
        self,
        symbol: str,
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
        *,
        force: bool,
        fetch_if_missing: bool,
        skip_enrichment: bool,
    ) -> pd.DataFrame:
        """Fetch or slice cached OHLCV covering the primary timestamp window."""
        cache = self._cache_path(symbol)
        cached: pd.DataFrame | None = None
        if cache.exists() and not force:
            cached = pd.read_parquet(cache)

        if cached is not None and self._cache_covers_window(
            cached, window_start, window_end, symbol
        ):
            out = cached.loc[cached.index <= window_end].copy()
            logger.info(
                "Loaded cached %s for primary window (%s -> %s, %d bars)",
                symbol,
                window_start,
                window_end,
                len(out),
            )
            return self._finalize_load(
                out, symbol, skip_enrichment=skip_enrichment, fetch_if_missing=False
            )

        if not fetch_if_missing and not force:
            have = 0 if cached is None else len(cached)
            raise RuntimeError(
                f"Cache for {symbol} does not cover {window_start} -> {window_end} "
                f"({have} bars at {cache}). Run download for this window first."
            )

        target_bars = self._bars_in_window(window_start, window_end)
        # When the cache already reaches the window start (or the exchange's earliest
        # candle for late-listed context symbols), only fetch the missing tail instead of
        # re-downloading the entire window to top up a few recent bars.
        fetch_start = window_start
        if (
            cached is not None
            and not cached.empty
            and self._window_start_covered(cached, window_start, symbol)
        ):
            tf = pd.Timedelta(minutes=timeframe_to_minutes(self.config.timeframe))
            fetch_start = max(window_start, cached.index.max() + tf)
        fetched, fetch_source = self._download(
            symbol,
            target_bars,
            since_ts=fetch_start,
            until_ts=window_end,
        )
        fetched = align_and_clean(fetched, self.config.timeframe)
        merged = self._merge_cached(cached, fetched)
        source = self._merged_provenance_source(cache, fetch_source)
        self._write_cache(merged, cache, symbol, source)
        out = merged.loc[merged.index <= window_end].copy()
        return self._finalize_load(out, symbol, skip_enrichment=skip_enrichment)

    def _window_start_covered(
        self, cached: pd.DataFrame, window_start: pd.Timestamp, symbol: str
    ) -> bool:
        """True when the cache begins early enough to cover the window start.

        Accepts a cache that starts at/before ``window_start`` or — for a symbol that
        listed after the window start — one that reaches the exchange's earliest available
        candle. Without the earliest-aware branch a late-listed context symbol (e.g. ETH
        or SOL perps, which list after BTC) could never satisfy the start check and would
        be fully re-downloaded on every enrichment pass.
        """
        if cached.empty:
            return False
        tf = pd.Timedelta(minutes=timeframe_to_minutes(self.config.timeframe))
        if cached.index.min() <= window_start + tf * 2:
            return True
        earliest = self._exchange_earliest_ts(symbol)
        return earliest is not None and cached.index.min() <= earliest + tf * 2

    def _cache_covers_window(
        self,
        cached: pd.DataFrame,
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
        symbol: str,
    ) -> bool:
        """True when cached context data spans the primary window.

        End coverage accepts a cache that reaches the window end or is otherwise live:
        context is forward-filled onto the primary grid, so a current cache a few bars
        behind the primary still covers the join causally and need not be re-fetched.
        """
        if cached.empty:
            return False
        if not self._window_start_covered(cached, window_start, symbol):
            return False
        tf = pd.Timedelta(minutes=timeframe_to_minutes(self.config.timeframe))
        return cached.index.max() >= window_end - tf * 2 or self._cache_is_live(cached)

    @staticmethod
    def _merge_cached(
        cached: pd.DataFrame | None,
        fetched: pd.DataFrame,
    ) -> pd.DataFrame:
        if cached is None or cached.empty:
            return fetched
        combined = pd.concat([cached, fetched])
        return combined[~combined.index.duplicated(keep="last")].sort_index()

    def _write_cache(
        self,
        df: pd.DataFrame,
        cache: Path,
        symbol: str,
        source: str,
    ) -> None:
        df.to_parquet(cache)
        write_data_provenance(
            cache,
            source=source,
            symbol=symbol,
            timeframe=self.config.timeframe,
            n_bars=len(df),
        )
        logger.info("Saved %d bars for %s to %s (%s)", len(df), symbol, cache, source)

    @staticmethod
    def _existing_provenance_source(cache: Path) -> str:
        meta = read_data_provenance(cache)
        if meta is None:
            return SOURCE_EXCHANGE
        return str(meta.get("source", SOURCE_EXCHANGE))

    @staticmethod
    def _merged_provenance_source(cache: Path, fetch_source: str) -> str:
        if fetch_source == SOURCE_EXCHANGE:
            return SOURCE_EXCHANGE
        return HistoricalDownloader._existing_provenance_source(cache)

    def _bars_in_window(
        self,
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
    ) -> int:
        minutes = (window_end - window_start).total_seconds() / 60.0
        return max(1, int(minutes / timeframe_to_minutes(self.config.timeframe)) + 2)

    def _cache_is_live(self, cached: pd.DataFrame) -> bool:
        """True when the cache includes bars near the current time."""
        if cached.empty:
            return False
        tf = pd.Timedelta(minutes=timeframe_to_minutes(self.config.timeframe))
        now = pd.Timestamp.now(tz="UTC")
        return cached.index.max() >= now - tf * 3

    def _cache_covers_start(self, cached: pd.DataFrame, symbol: str) -> bool:
        """True when cached history reaches as far back as we can actually fetch.

        The cache "covers start" when it begins near the configured start date, **or**
        — when the configured start predates the symbol's exchange listing — when it
        already reaches the exchange's earliest available candle. The latter probe
        prevents pointlessly re-backfilling the whole history on every run for symbols
        whose data simply does not extend back to ``historical_start_date`` (e.g. an
        ``earliest`` start that resolves to 2017 while binanceusdm futures begin 2019).
        """
        if cached.empty:
            return False
        start = datetime.fromisoformat(self.config.data.start_date_iso()).replace(tzinfo=UTC)
        tf = pd.Timedelta(minutes=timeframe_to_minutes(self.config.timeframe))
        if cached.index.min() <= pd.Timestamp(start) + tf * 2:
            return True
        # Cache starts later than the configured start: it may already hold all the
        # history the exchange offers. Probe the earliest candle and accept the cache
        # when it reaches that bar; on probe failure keep the conservative answer.
        earliest = self._exchange_earliest_ts(symbol)
        if earliest is None:
            return False
        return cached.index.min() <= earliest + tf * 2

    def _exchange_earliest_ts(self, symbol: str) -> pd.Timestamp | None:
        """Probe the exchange's earliest available candle timestamp (memoized).

        Returns ``None`` when ccxt is unavailable or the exchange cannot be reached, so
        callers fall back to conservative behaviour. The result is cached per symbol for
        the lifetime of this downloader to avoid repeat probes.
        """
        if symbol in self._earliest_ts_cache:
            return self._earliest_ts_cache[symbol]
        earliest: pd.Timestamp | None = None
        try:
            import ccxt  # noqa: PLC0415 - optional dependency, imported lazily

            exchange_cls = getattr(ccxt, self.config.data.exchange, None)
            if exchange_cls is not None:
                exchange = exchange_cls({"enableRateLimit": True})
                since = self._start_since_ms(exchange)
                batch = exchange.fetch_ohlcv(
                    symbol, timeframe=self.config.timeframe, since=since, limit=1
                )
                if batch:
                    earliest = pd.to_datetime(batch[0][0], unit="ms", utc=True)
                    logger.info(
                        "Exchange earliest %s candle (%s) is %s.",
                        symbol,
                        self.config.timeframe,
                        earliest,
                    )
        except Exception as exc:  # noqa: BLE001 - probe is best-effort context
            logger.info("Could not probe earliest %s candle: %s", symbol, exc)
        self._earliest_ts_cache[symbol] = earliest
        return earliest

    def _since_ms_for_recent_bars(self, exchange, target_bars: int) -> int:
        """Estimate ``since`` so paginating forward yields roughly the latest ``target_bars``."""
        tf_ms = timeframe_to_minutes(self.config.timeframe) * 60_000
        lookback_ms = int(target_bars * tf_ms * 1.2)
        now_ms = int(datetime.now(UTC).timestamp() * 1000)
        return max(self._start_since_ms(exchange), now_ms - lookback_ms)

    def _finalize_load(
        self,
        df: pd.DataFrame,
        symbol: str,
        *,
        skip_enrichment: bool,
        fetch_if_missing: bool = True,
    ) -> pd.DataFrame:
        """Optionally enrich the primary symbol with cross-asset and alt data."""
        if skip_enrichment or symbol != self.config.primary_symbol:
            return df
        from epoch_ai.data.enrichment import enrich_primary_market

        return enrich_primary_market(
            df, self.config, self, fetch_if_missing=fetch_if_missing
        )

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
        since_ts: pd.Timestamp | None = None,
        until_ts: pd.Timestamp | None = None,
        recent_tail: bool = False,
    ) -> tuple[pd.DataFrame, str]:
        """Try CCXT first; fall back to synthetic data on any failure."""
        windowed = since_ts is not None or until_ts is not None
        base_len = len(base_df) if base_df is not None else 0
        if not windowed and not recent_tail and base_len >= target_bars:
            return base_df.iloc[:target_bars].copy(), SOURCE_EXCHANGE  # type: ignore[union-attr]

        ccxt_reason: str | None = None
        try:
            df = self._download_ccxt(
                symbol,
                target_bars,
                base_df=base_df,
                since_ts=since_ts,
                until_ts=until_ts,
                recent_tail=recent_tail,
            )
            if df is not None and len(df) > 0:
                return df, SOURCE_EXCHANGE
            ccxt_reason = "CCXT returned no data"
        except Exception as exc:  # noqa: BLE001 - any failure should fall back
            ccxt_reason = f"CCXT download failed ({exc})"

        if not self.config.data.use_synthetic_fallback:
            detail = ccxt_reason or "CCXT unavailable"
            if base_df is not None and len(base_df) > 0:
                logger.warning(
                    "CCXT unavailable for %s (%s); using %d cached real bars (target %d).",
                    symbol,
                    detail,
                    len(base_df),
                    target_bars,
                )
                return base_df, SOURCE_EXCHANGE
            raise RuntimeError(
                f"No data available for {symbol} and synthetic fallback disabled ({detail}). "
                "Install ccxt (requirements-optional.txt) and download real exchange data, "
                "or provide a provenanced parquet cache under data.data_dir."
            )

        if base_df is not None and len(base_df) > 0:
            logger.warning(
                "%s for %s; keeping %d cached bars (target %d).",
                ccxt_reason,
                symbol,
                len(base_df),
                target_bars,
            )
            return base_df, self._existing_provenance_source(
                self._cache_path(symbol)
            )

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
        return df, SOURCE_SYNTHETIC

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
        since_ts: pd.Timestamp | None = None,
        until_ts: pd.Timestamp | None = None,
        recent_tail: bool = False,
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

        if since_ts is not None:
            since = int(since_ts.timestamp() * 1000)
            desc = f"Downloading {symbol}"
            partial = None
        elif base_df is not None and not base_df.empty:
            since = int(base_df.index.max().timestamp() * 1000) + tf_ms
            desc = f"Extending {symbol}"
            partial: pd.DataFrame | None = base_df.copy()
        else:
            since = (
                self._since_ms_for_recent_bars(exchange, target_bars)
                if recent_tail
                else self._start_since_ms(exchange)
            )
            desc = f"Downloading {symbol}"
            partial = None

        limit = 1000
        with DownloadProgressBar(total=target_bars, desc=desc) as progress:
            start = len(partial) if partial is not None else 0
            progress.advance_to(start, render=False)
            progress.begin_rate_tracking()
            progress.refresh()

            while True:
                if until_ts is not None and partial is not None:
                    # Human: do not stop just because max >= until when the exchange's
                    # first listing is entirely after the window (empty slice + NaT).
                    if partial.index.min() > until_ts:
                        logger.info(
                            "Earliest %s bar (%s) is after primary window end (%s); "
                            "no overlapping history for this slice.",
                            symbol,
                            partial.index.min(),
                            until_ts,
                        )
                        return None
                    if partial.index.min() <= until_ts and partial.index.max() >= until_ts:
                        break
                elif until_ts is None and partial is not None and len(partial) >= target_bars and not recent_tail:
                    break
                elif recent_tail and partial is not None:
                    now = pd.Timestamp.now(tz="UTC")
                    tf = pd.Timedelta(minutes=timeframe_to_minutes(timeframe))
                    if partial.index.max() >= now - tf * 2:
                        break

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
                if until_ts is None and len(batch) < limit:
                    break
                if until_ts is not None and len(batch) < limit:
                    break

            final_len = 0 if partial is None else len(partial)
            if until_ts is None and final_len < target_bars:
                progress.set_total(max(final_len, 1))

        if partial is None or partial.empty:
            return None

        if final_len < target_bars and until_ts is None:
            logger.info(
                "Exchange history ends at %d bars for %s (requested %d).",
                final_len,
                symbol,
                target_bars,
            )

        if until_ts is not None and since_ts is not None:
            df = partial.loc[(partial.index >= since_ts) & (partial.index <= until_ts)].copy()
        elif until_ts is not None:
            df = partial.loc[partial.index <= until_ts].copy()
        elif recent_tail:
            df = partial.iloc[-target_bars:].copy()
        else:
            df = partial.iloc[:target_bars].copy()

        if df.empty:
            logger.info(
                "No %s bars in primary window (%s -> %s).",
                symbol,
                since_ts or partial.index.min(),
                until_ts or partial.index.max(),
            )
            return None

        df = self._attach_funding(exchange, symbol, df)
        if self.config.data.fetch_open_interest:
            df = self._attach_open_interest(exchange, symbol, df)
        return df

    def _attach_funding(self, exchange, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """Paginate funding-rate history for derivatives markets."""
        if df.empty:
            return df
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
        if df.empty:
            return df
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

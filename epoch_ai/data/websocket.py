"""Real-time WebSocket data handler (live / near-real-time mode).

This wraps ``ccxt.pro`` (when installed) to stream live candles for the configured
symbols and maintain a rolling in-memory OHLCV buffer that the live trading loop can
consume. It is intentionally lightweight; full order-book depth streaming can be
layered on later via the same pattern.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable

import pandas as pd

from epoch_ai.config.settings import AppConfig
from epoch_ai.utils.logging import get_logger

logger = get_logger(__name__)


class RealtimeDataHandler:
    """Maintain a rolling OHLCV buffer fed by an exchange WebSocket."""

    def __init__(self, config: AppConfig, buffer_size: int = 5000) -> None:
        self.config = config
        self.buffer_size = buffer_size
        self._buffers: dict[str, deque] = {
            symbol: deque(maxlen=buffer_size) for symbol in config.symbols
        }
        self._last_ts: dict[str, int | None] = dict.fromkeys(config.symbols)

    def get_frame(self, symbol: str) -> pd.DataFrame:
        """Return the current rolling buffer for ``symbol`` as a DataFrame."""
        rows = list(self._buffers[symbol])
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df.drop(columns="ts").set_index("timestamp")

    def ingest_candle(self, symbol: str, candle: list) -> bool:
        """Append one OHLCV candle if it is new; return whether the buffer changed."""
        ts = int(candle[0])
        if self._last_ts.get(symbol) == ts:
            return False
        self._buffers[symbol].append(candle)
        self._last_ts[symbol] = ts
        return True

    async def stream(
        self,
        on_candle: Callable[[str, pd.DataFrame], None] | None = None,
    ) -> None:
        """Stream candles via ``ccxt.pro`` until cancelled.

        Args:
            on_candle: Optional callback invoked with ``(symbol, frame)`` whenever a
                new candle timestamp is observed.

        Raises:
            RuntimeError: If ``ccxt.pro`` is not available.
        """
        try:
            import ccxt.pro as ccxtpro  # noqa: PLC0415 - optional dependency
        except ImportError as exc:  # pragma: no cover - requires optional dep
            raise RuntimeError(
                "ccxt.pro is required for live streaming. Install with "
                "`pip install -r requirements-optional.txt`."
            ) from exc

        exchange_cls = getattr(ccxtpro, self.config.data.exchange)
        exchange = exchange_cls({"enableRateLimit": True})
        try:
            while True:
                for symbol in self.config.symbols:
                    candles = await exchange.watch_ohlcv(symbol, self.config.timeframe)
                    for candle in candles:
                        if self.ingest_candle(symbol, candle) and on_candle is not None:
                            on_candle(symbol, self.get_frame(symbol))
        except asyncio.CancelledError:  # pragma: no cover - cooperative shutdown
            logger.info("WebSocket stream cancelled; closing exchange.")
        finally:  # pragma: no cover - network teardown
            await exchange.close()

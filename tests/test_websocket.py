"""Tests for WebSocket buffer handler (no network)."""

from __future__ import annotations

from epoch_ai.config.settings import AppConfig
from epoch_ai.data.websocket import RealtimeDataHandler


def test_ingest_candle_deduplicates():
    handler = RealtimeDataHandler(AppConfig())
    candle = [1_700_000_000_000, 1.0, 2.0, 0.5, 1.5, 100.0]
    assert handler.ingest_candle("BTC/USDT", candle) is True
    assert handler.ingest_candle("BTC/USDT", candle) is False
    frame = handler.get_frame("BTC/USDT")
    assert len(frame) == 1
    assert float(frame["close"].iloc[0]) == 1.5

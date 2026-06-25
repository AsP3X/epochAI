"""Data layer: historical downloads, synthetic fallback, cleaning, live feeds."""

from __future__ import annotations

from epoch_ai.data.cleaning import align_and_clean
from epoch_ai.data.downloader import HistoricalDownloader
from epoch_ai.data.synthetic import generate_synthetic_ohlcv
from epoch_ai.data.websocket import RealtimeDataHandler

__all__ = [
    "HistoricalDownloader",
    "RealtimeDataHandler",
    "align_and_clean",
    "generate_synthetic_ohlcv",
]

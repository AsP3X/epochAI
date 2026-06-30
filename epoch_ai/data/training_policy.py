"""Real-data policy for supervised training paths."""

from __future__ import annotations

from epoch_ai.config.settings import AppConfig
from epoch_ai.data.downloader import HistoricalDownloader
from epoch_ai.data.provenance import assert_cache_is_real
from epoch_ai.utils.logging import get_logger

logger = get_logger(__name__)


def config_for_supervised_training(config: AppConfig) -> AppConfig:
    """Return a copy with synthetic fallback disabled for all backends."""
    if not config.data.use_synthetic_fallback:
        return config
    cfg = config.model_copy(deep=True)
    cfg.data.use_synthetic_fallback = False
    logger.info(
        "Supervised training disables synthetic fallback (real exchange or provenanced cache only)."
    )
    return cfg


def assert_training_cache_real(config: AppConfig, symbol: str | None = None) -> None:
    """Verify the parquet cache for ``symbol`` is exchange-sourced when cached."""
    sym = symbol or config.primary_symbol
    downloader = HistoricalDownloader(config)
    cache_path = downloader._cache_path(sym)
    if not cache_path.exists():
        # Patched/offline test downloads may return in-memory frames without persisting.
        return
    assert_cache_is_real(cache_path, symbol=sym)

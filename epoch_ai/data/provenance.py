"""Parquet cache provenance — distinguish exchange data from synthetic fallback.

Training, retrain, and promotion paths require ``SOURCE_EXCHANGE`` caches.
Synthetic data remains available for unit tests and explicit offline demos only.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from epoch_ai.utils.logging import get_logger

logger = get_logger(__name__)

SOURCE_EXCHANGE = "exchange"
SOURCE_SYNTHETIC = "synthetic"


def provenance_path(cache_path: Path) -> Path:
    """Sidecar JSON path for a parquet cache file."""
    return cache_path.with_suffix(".provenance.json")


def write_data_provenance(
    cache_path: Path,
    *,
    source: str,
    symbol: str,
    timeframe: str,
    n_bars: int,
) -> None:
    """Persist cache lineage next to the parquet file."""
    payload: dict[str, Any] = {
        "source": source,
        "symbol": symbol,
        "timeframe": timeframe,
        "n_bars": n_bars,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    path = provenance_path(cache_path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.debug("Wrote data provenance %s (%s, %d bars).", path, source, n_bars)


def read_data_provenance(cache_path: Path) -> dict[str, Any] | None:
    """Return provenance metadata when the sidecar exists."""
    path = provenance_path(cache_path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read provenance %s: %s", path, exc)
        return None


def assert_cache_is_real(cache_path: Path, *, symbol: str) -> None:
    """Raise when a cache is synthetic or lacks exchange provenance.

    Args:
        cache_path: Parquet cache path for ``symbol``.
        symbol: Human-readable pair for error messages.

    Raises:
        RuntimeError: Cache is missing, unprovenanced, or marked synthetic.
    """
    if not cache_path.exists():
        raise RuntimeError(
            f"No cached data for {symbol} at {cache_path}. "
            f"Run: python -m epoch_ai download --full-history"
        )
    meta = read_data_provenance(cache_path)
    if meta is None:
        raise RuntimeError(
            f"Cache for {symbol} at {cache_path} has no provenance metadata "
            "(legacy or unknown origin). Training requires verified exchange data. "
            f"Re-download: python -m epoch_ai download --full-history --force"
        )
    source = str(meta.get("source", ""))
    if source == SOURCE_SYNTHETIC:
        raise RuntimeError(
            f"Cache for {symbol} at {cache_path} is marked synthetic. "
            "Training requires real exchange data. Delete the cache and run: "
            f"python -m epoch_ai download --full-history"
        )
    if source != SOURCE_EXCHANGE:
        raise RuntimeError(
            f"Cache for {symbol} has unknown provenance source {source!r}. "
            f"Re-download: python -m epoch_ai download --full-history --force"
        )

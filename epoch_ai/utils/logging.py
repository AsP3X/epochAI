"""Application logging helpers (distinct from the prediction/outcome log store)."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def setup_logging(level: int | str = logging.INFO) -> None:
    """Configure root logging once with a concise, readable format.

    Args:
        level: Logging level (name or numeric).
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, ensuring logging is configured first."""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)

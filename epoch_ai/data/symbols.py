"""Symbol naming helpers for cross-asset joins."""

from __future__ import annotations


def asset_prefix(symbol: str) -> str:
    """Return a lowercase asset prefix from a CCXT symbol (``ETH/USDT`` -> ``eth``)."""
    base = symbol.split(":")[0].split("/")[0]
    return base.lower()

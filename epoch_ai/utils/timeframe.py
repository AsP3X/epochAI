"""Helpers for converting exchange timeframe strings to durations/frequencies."""

from __future__ import annotations

import math

_UNIT_MINUTES = {"m": 1, "h": 60, "d": 60 * 24, "w": 60 * 24 * 7}

# pandas frequency aliases keyed by the timeframe unit suffix.
_UNIT_PANDAS = {"m": "min", "h": "h", "d": "D", "w": "W"}

_MINUTES_PER_YEAR = 365 * 24 * 60


def _split(timeframe: str) -> tuple[int, str]:
    timeframe = timeframe.strip().lower()
    if len(timeframe) < 2 or not timeframe[:-1].isdigit():
        raise ValueError(f"Unrecognised timeframe: {timeframe!r}")
    return int(timeframe[:-1]), timeframe[-1]


def timeframe_to_minutes(timeframe: str) -> int:
    """Convert a timeframe like ``"15m"``/``"4h"``/``"1d"`` to minutes."""
    value, unit = _split(timeframe)
    if unit not in _UNIT_MINUTES:
        raise ValueError(f"Unsupported timeframe unit: {unit!r}")
    return value * _UNIT_MINUTES[unit]


def timeframe_to_pandas_freq(timeframe: str) -> str:
    """Convert a timeframe to a pandas offset alias (e.g. ``"15m" -> "15min"``)."""
    value, unit = _split(timeframe)
    if unit not in _UNIT_PANDAS:
        raise ValueError(f"Unsupported timeframe unit: {unit!r}")
    return f"{value}{_UNIT_PANDAS[unit]}"


def annualization_factor(timeframe: str) -> float:
    """Return the sqrt-annualization factor for Sharpe-style metrics.

    This is ``sqrt(periods_per_year)`` for the given bar size.
    """
    minutes = timeframe_to_minutes(timeframe)
    periods_per_year = _MINUTES_PER_YEAR / minutes
    return math.sqrt(periods_per_year)

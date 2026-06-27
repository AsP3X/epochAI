"""Crypto Fear & Greed index data source (alternative.me).

The Fear & Greed index is a free, **daily**, market-wide sentiment gauge in ``[0, 100]``
(0 = extreme fear, 100 = extreme greed) with history back to **February 2018** — long
enough to cover essentially all of the BTC perpetual-futures history this project trains
on. It is orthogonal to raw price action and is consumed by
:class:`~epoch_ai.features.sentiment.SentimentFeatures` once joined onto the OHLCV frame
as a ``fear_greed`` column.

Network access is optional. Any failure (offline, geo-block, schema change) returns
``None`` so the pipeline degrades gracefully to "no sentiment column" — exactly like the
other optional data sources (funding, on-chain). No third-party dependency is required;
the public JSON endpoint is read with the standard library.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.request import Request, urlopen

import pandas as pd

from epoch_ai.utils.logging import get_logger

logger = get_logger(__name__)

# ``limit=0`` asks alternative.me for the full available history.
FNG_URL = "https://api.alternative.me/fng/?limit=0&format=json"


def fetch_fear_greed(*, url: str = FNG_URL, timeout: float = 15.0) -> pd.Series | None:
    """Fetch the full Fear & Greed history as a daily UTC Series.

    Args:
        url: Override the alternative.me endpoint (used by tests).
        timeout: Socket timeout in seconds for the HTTP request.

    Returns:
        A Series named ``fear_greed`` (float in ``[0, 100]``) indexed by tz-aware UTC
        timestamps and sorted ascending, or ``None`` when the index cannot be retrieved
        or parsed (so callers can degrade gracefully).
    """
    try:
        request = Request(url, headers={"User-Agent": "epoch-ai/1.0"})
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed https host
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - sentiment is optional, degrade gracefully
        logger.info("Fear & Greed index unavailable: %s", exc)
        return None

    series = parse_fear_greed(payload)
    if series is None:
        logger.info("Fear & Greed payload contained no usable records.")
    else:
        logger.info(
            "Fetched %d Fear & Greed readings (%s -> %s).",
            len(series),
            series.index[0].date(),
            series.index[-1].date(),
        )
    return series


def parse_fear_greed(payload: Any) -> pd.Series | None:
    """Parse an alternative.me FNG JSON payload into a daily ``fear_greed`` Series.

    The payload's ``data`` list holds ``{"value": "40", "timestamp": "1517463000", ...}``
    records whose ``timestamp`` is epoch-seconds (UTC) for the reading. Malformed
    records are skipped rather than aborting the whole parse.
    """
    data = payload.get("data") if isinstance(payload, dict) else None
    if not data:
        return None

    timestamps: list[pd.Timestamp] = []
    values: list[float] = []
    for item in data:
        try:
            ts = pd.to_datetime(int(item["timestamp"]), unit="s", utc=True)
            value = float(item["value"])
        except (KeyError, TypeError, ValueError):
            continue
        timestamps.append(ts)
        values.append(value)

    if not timestamps:
        return None

    series = pd.Series(values, index=pd.DatetimeIndex(timestamps), name="fear_greed")
    series = series[~series.index.duplicated(keep="last")].sort_index()
    series.index.name = "timestamp"
    return series

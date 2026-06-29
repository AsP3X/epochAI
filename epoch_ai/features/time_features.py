"""Time-based and cyclical calendar features.

Crypto markets exhibit intraday and weekly seasonality (e.g. funding times, regional
trading sessions). These are encoded with sine/cosine pairs so the model sees them as
smooth cyclical signals rather than arbitrary integers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from epoch_ai.features.base import FeatureGroup


def _cyclical(values: pd.Series, period: int, name: str) -> pd.DataFrame:
    radians = 2.0 * np.pi * values / period
    return pd.DataFrame(
        {f"time_{name}_sin": np.sin(radians), f"time_{name}_cos": np.cos(radians)},
        index=values.index,
    )


class TimeFeatures(FeatureGroup):
    """Cyclical encodings for hour-of-day, day-of-week and day-of-month."""

    name = "time"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            raise TypeError("TimeFeatures requires a DatetimeIndex.")

        parts = [
            _cyclical(pd.Series(idx.hour, index=idx), 24, "hour"),
            _cyclical(pd.Series(idx.dayofweek, index=idx), 7, "dow"),
            _cyclical(pd.Series(idx.day, index=idx), 31, "dom"),
            _cyclical(pd.Series(idx.month, index=idx), 12, "month"),
        ]
        out = pd.concat(parts, axis=1)
        out["time_is_weekend"] = (idx.dayofweek >= 5).astype(float)

        hour = idx.hour
        out["time_session_asia"] = ((hour >= 0) & (hour < 8)).astype(float)
        out["time_session_europe"] = ((hour >= 7) & (hour < 16)).astype(float)
        out["time_session_us"] = ((hour >= 13) & (hour < 22)).astype(float)

        # 8h funding cadence on Binance USDT-M
        minutes = idx.hour * 60 + idx.minute
        funding_minutes = np.array([0, 480, 960])  # 00:00, 08:00, 16:00 UTC
        dist = np.min(
            np.abs(minutes.to_numpy()[:, None] - funding_minutes[None, :]),
            axis=1,
        )
        out["time_minutes_to_funding"] = dist / 480.0
        out["time_is_funding_hour"] = (dist <= 15).astype(float)

        dom = idx.day
        out["time_quarter_end"] = dom.isin([28, 29, 30, 31]).astype(float)

        return out

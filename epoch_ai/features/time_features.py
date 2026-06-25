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
        ]
        out = pd.concat(parts, axis=1)
        out["time_is_weekend"] = (idx.dayofweek >= 5).astype(float)
        return out

"""Optional sentiment feature group.

Integration point for external alternative data (e.g. the Fear & Greed index, social
volume). It degrades gracefully: when a ``fear_greed`` column (or ``social_volume``)
is present on the frame it derives causal features from it; otherwise it returns no
columns so enabling the group never breaks the pipeline.

All transforms use only past-or-current values (rolling/diff), preserving causality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from epoch_ai.features.base import FeatureGroup
from epoch_ai.utils.logging import get_logger

logger = get_logger(__name__)


class SentimentFeatures(FeatureGroup):
    """External sentiment signals (Fear & Greed, social volume) when available."""

    name = "sent"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        emitted = False

        if "fear_greed" in df.columns:
            fg = df["fear_greed"]
            # Normalise the 0-100 index to [-1, 1] and capture its momentum.
            out["sent_fear_greed"] = fg / 50.0 - 1.0
            out["sent_fear_greed_chg"] = fg.diff()
            out["sent_fear_greed_z"] = (
                fg - fg.rolling(96, min_periods=16).mean()
            ) / fg.rolling(96, min_periods=16).std().replace(0.0, np.nan)
            emitted = True

        if "social_volume" in df.columns:
            sv = df["social_volume"]
            sv_ma = sv.rolling(96, min_periods=16).mean()
            out["sent_social_dist"] = sv / sv_ma.replace(0.0, np.nan) - 1.0
            emitted = True

        if not emitted:
            logger.info(
                "SentimentFeatures found no sentiment columns (e.g. 'fear_greed', "
                "'social_volume'); returning no columns. Join a data source to activate."
            )
        return out

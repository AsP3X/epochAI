"""Optional sentiment / on-chain feature group (skeleton).

This is a placeholder that documents the integration point for external alternative
data (e.g. Fear & Greed index, social volume, on-chain flows). It is disabled by
default. When no external source is wired in it returns an empty frame, so enabling
it never breaks the pipeline; replace :meth:`compute` with real data joins.
"""

from __future__ import annotations

import pandas as pd

from epoch_ai.features.base import FeatureGroup
from epoch_ai.utils.logging import get_logger

logger = get_logger(__name__)


class SentimentFeatures(FeatureGroup):
    """Placeholder for external sentiment/on-chain signals."""

    name = "sent"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(
            "SentimentFeatures is a stub returning no columns. Wire in an external "
            "data source (e.g. Fear & Greed, social volume) to activate it."
        )
        return pd.DataFrame(index=df.index)

"""Feature pipeline: assemble feature groups and build supervised targets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from epoch_ai.config.settings import AppConfig, PredictionConfig
from epoch_ai.features.base import FeatureGroup, build_feature_groups
from epoch_ai.utils.logging import get_logger

logger = get_logger(__name__)


class FeaturePipeline:
    """Compute the full engineered feature matrix from raw market data.

    The pipeline runs every enabled :class:`FeatureGroup` and concatenates their
    outputs into a single, column-stable feature matrix. Because each group is
    causal, any row of the resulting matrix is a valid feature vector for predicting
    the *future* - which is exactly what the progressive walk-forward engine stores
    at prediction time.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.groups: list[FeatureGroup] = build_feature_groups(
            config.features,
            context_symbols=config.data.context_symbols,
        )
        self._feature_names: list[str] | None = None

    @property
    def feature_names(self) -> list[str]:
        """Ordered feature-column names (available after :meth:`transform`)."""
        if self._feature_names is None:
            raise RuntimeError("Call transform() before accessing feature_names.")
        return self._feature_names

    def transform(self, df: pd.DataFrame, *, log_stats: bool = True) -> pd.DataFrame:
        """Compute and concatenate features for all enabled groups.

        Args:
            df: Cleaned OHLCV(+context) frame indexed by ``timestamp``.
            log_stats: When ``False``, skip per-call INFO logs (used on live ticks).

        Returns:
            Feature matrix indexed by ``timestamp``; infinite values are replaced
            with NaN and (optionally) rows with any NaN are dropped per config.
        """
        if df.empty:
            raise ValueError("Cannot compute features on an empty frame.")

        if self.config.data.synthesize_market_extensions:
            from epoch_ai.data.market_extensions import extend_market_columns

            df = extend_market_columns(df, seed=self.config.data.synthetic_seed)

        frames = [group.compute(df) for group in self.groups]
        if not frames:
            raise ValueError("No feature groups enabled; enable at least one.")

        features = pd.concat(frames, axis=1)
        features = features.replace([np.inf, -np.inf], np.nan)
        if log_stats:
            logger.info(
                "Computed %d features across %d groups for %d bars.",
                features.shape[1],
                len(self.groups),
                len(features),
            )

        if self.config.features.dropna:
            before = len(features)
            features = features.dropna()
            if log_stats:
                logger.info("Dropped %d warm-up rows with NaN features.", before - len(features))

        self._feature_names = list(features.columns)
        return features


def build_target(df: pd.DataFrame, prediction: PredictionConfig) -> pd.Series:
    """Build the supervised target aligned to each bar's *entry* time.

    The target at time ``t`` describes the forward move realised between ``t`` and
    ``t + horizon`` bars. The final ``horizon`` rows have no realised future and are
    therefore ``NaN`` (they become live, not-yet-resolved predictions).

    Args:
        df: Cleaned OHLCV frame.
        prediction: Prediction/target configuration.

    Returns:
        A Series named ``target`` aligned to ``df.index``.
    """
    horizon = prediction.horizon
    forward_return = df["close"].shift(-horizon) / df["close"] - 1.0

    if prediction.task == "regression":
        target = forward_return
    elif prediction.neutral_band > 0.0:
        # Dead-zone labelling: only decisive moves become training labels; bars inside
        # the band (and the unresolved final ``horizon`` rows) stay NaN and are dropped
        # downstream, so the model is not trained on near-zero directional noise.
        upper = prediction.threshold + prediction.neutral_band
        lower = prediction.threshold - prediction.neutral_band
        target = pd.Series(np.nan, index=df.index, name="target")
        target[forward_return > upper] = 1.0
        target[forward_return < lower] = 0.0
    else:
        target = (forward_return > prediction.threshold).astype(float)
        target[forward_return.isna()] = np.nan

    target.name = "target"
    return target


def forward_return(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Return the realised forward return over ``horizon`` bars (for outcome logs)."""
    fr = df["close"].shift(-horizon) / df["close"] - 1.0
    fr.name = "forward_return"
    return fr

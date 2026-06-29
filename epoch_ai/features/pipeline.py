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


def build_multi_horizon_targets(
    df: pd.DataFrame, prediction: PredictionConfig
) -> pd.DataFrame:
    """Build per-horizon forward-return and direction targets aligned to each bar.

    For each horizon ``h`` in ``prediction.horizons`` the frame contains:

    - ``ret_{h}``: forward log return ``log(close[t+h] / close[t])`` (quantile target).
    - ``target_{h}``: classification label or regression return for that horizon.

    The final ``h`` rows per horizon are ``NaN`` (unresolved future).

    Args:
        df: Cleaned OHLCV frame.
        prediction: Prediction/target configuration.

    Returns:
        DataFrame indexed like ``df`` with ``ret_*`` and ``target_*`` columns.
    """
    close = df["close"]
    out = pd.DataFrame(index=df.index)
    for h in prediction.horizons:
        simple_ret = close.shift(-h) / close - 1.0
        log_ret = np.log(close.shift(-h) / close)
        out[f"ret_{h}"] = log_ret
        if prediction.task == "regression":
            out[f"target_{h}"] = log_ret
        elif prediction.neutral_band > 0.0:
            upper = prediction.threshold + prediction.neutral_band
            lower = prediction.threshold - prediction.neutral_band
            label = pd.Series(np.nan, index=df.index)
            label[simple_ret > upper] = 1.0
            label[simple_ret < lower] = 0.0
            out[f"target_{h}"] = label
        else:
            label = (simple_ret > prediction.threshold).astype(float)
            label[simple_ret.isna()] = np.nan
            out[f"target_{h}"] = label
    return out


def build_target(df: pd.DataFrame, prediction: PredictionConfig) -> pd.Series:
    """Build the primary supervised target (backward-compatible single Series).

    Returns the ``target_{horizon}`` column from :func:`build_multi_horizon_targets`
    for ``prediction.horizon``, named ``target`` for legacy consumers.

    Args:
        df: Cleaned OHLCV frame.
        prediction: Prediction/target configuration.

    Returns:
        A Series named ``target`` aligned to ``df.index``.
    """
    targets = build_multi_horizon_targets(df, prediction)
    target = targets[f"target_{prediction.horizon}"].copy()
    target.name = "target"
    return target


def forward_return(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Return the realised forward return over ``horizon`` bars (for outcome logs)."""
    fr = df["close"].shift(-horizon) / df["close"] - 1.0
    fr.name = "forward_return"
    return fr

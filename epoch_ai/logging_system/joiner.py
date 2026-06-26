"""Join predictions with realised outcomes to build training datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from epoch_ai.logging_system.store import PredictionStore


@dataclass(frozen=True, slots=True)
class RetrainLogStats:
    """Counts from the SQLite store relevant to ``retrain --min-new-samples``."""

    predictions: int
    outcomes: int
    joined_samples: int
    pending: int

    @property
    def max_min_new_samples(self) -> int:
        """Maximum ``--min-new-samples`` that still uses the SQLite log path."""
        return self.joined_samples


def join_predictions_outcomes(store: PredictionStore, symbol: str | None = None) -> pd.DataFrame:
    """Join logged predictions with their realised outcomes.

    Args:
        store: An open :class:`PredictionStore`.
        symbol: Optional symbol filter.

    Returns:
        A DataFrame with one row per resolved prediction, including the prediction,
        confidence, realised return/label and parsed context.
    """
    preds = store.predictions_frame(symbol)
    outs = store.outcomes_frame()
    if preds.empty or outs.empty:
        return pd.DataFrame()

    merged = preds.merge(
        outs, left_on="id", right_on="prediction_id", suffixes=("", "_outcome")
    )
    return merged


def build_training_dataset(store: PredictionStore, symbol: str | None = None) -> pd.DataFrame:
    """Reconstruct a feature matrix + realised label from logged history.

    This demonstrates the "predictions + outcomes -> training data" loop: the stored
    feature vectors become ``X`` and the realised labels become ``y`` (with the
    realised forward return and context retained for analysis / recency weighting).

    Returns:
        A DataFrame whose columns are the original features plus ``target``,
        ``forward_return`` and ``timestamp``.
    """
    merged = join_predictions_outcomes(store, symbol)
    if merged.empty:
        return pd.DataFrame()

    feature_rows = [json.loads(f) for f in merged["features"]]
    features = pd.DataFrame(feature_rows, index=merged.index)
    features["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True)
    features["forward_return"] = merged["forward_return"].to_numpy()
    features["target"] = merged["realized_label"].to_numpy()
    return features


def retrain_log_stats(store: PredictionStore, symbol: str | None = None) -> RetrainLogStats:
    """Summarise how many rows are available for ``retrain --min-new-samples``.

    ``joined_samples`` is what :func:`run_retrain` compares against ``min_new_samples``.
    Predictions in the last ``horizon`` bars of a session may stay ``pending`` until
    the forward window elapses.
    """
    preds = store.predictions_frame(symbol)
    if preds.empty:
        return RetrainLogStats(predictions=0, outcomes=0, joined_samples=0, pending=0)

    outs = store.outcomes_frame()
    resolved_ids = set(outs["prediction_id"]) if not outs.empty else set()
    pending = int((~preds["id"].isin(resolved_ids)).sum())
    joined = len(build_training_dataset(store, symbol))
    return RetrainLogStats(
        predictions=len(preds),
        outcomes=joined,
        joined_samples=joined,
        pending=pending,
    )

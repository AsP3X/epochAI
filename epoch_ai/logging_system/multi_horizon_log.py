"""Helpers for logging and resolving per-horizon prediction rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from epoch_ai.logging_system.schemas import OutcomeLog
from epoch_ai.logging_system.store import PredictionStore
from epoch_ai.services.types import MultiHorizonPredictionResult

if TYPE_CHECKING:
    from epoch_ai.calibration.tracker import CalibrationTracker


@dataclass(slots=True)
class PendingHorizonLog:
    """One logged horizon row awaiting outcome resolution."""

    prediction_id: int
    entry_index: int
    entry_price: float
    horizon: int
    raw_prediction: float


def log_multi_horizon_bar(
    store: PredictionStore,
    result: MultiHorizonPredictionResult,
    *,
    signal: int,
    base_features: dict[str, float],
    entry_price: float,
    entry_index: int,
) -> list[PendingHorizonLog]:
    """Persist one SQLite row per horizon and return pending outcome trackers."""
    ids = store.log_multi_horizon(
        result,
        base_features=base_features,
        entry_price=entry_price,
        signal=signal,
    )
    pending: list[PendingHorizonLog] = []
    for forecast, pred_id in zip(result.horizons, ids, strict=True):
        pending.append(
            PendingHorizonLog(
                prediction_id=pred_id,
                entry_index=entry_index,
                entry_price=entry_price,
                horizon=forecast.horizon,
                raw_prediction=forecast.p_up,
            )
        )
    return pending


def resolve_pending_horizons(
    pending: list[PendingHorizonLog],
    *,
    current_index: int,
    close: pd.Series,
    index: pd.Index,
    threshold: float,
    store: PredictionStore,
    calibration: CalibrationTracker | None = None,
    context: dict[str, Any] | None = None,
) -> list[PendingHorizonLog]:
    """Resolve matured horizon rows and drop them from the pending queue."""
    still_pending: list[PendingHorizonLog] = []
    ctx = dict(context or {})

    for item in pending:
        if current_index - item.entry_index < item.horizon:
            still_pending.append(item)
            continue
        resolve_index = min(item.entry_index + item.horizon, current_index)
        resolve_ts = index[resolve_index]
        exit_price = float(close.loc[resolve_ts])
        forward_return = exit_price / item.entry_price - 1.0
        realized_label = int(forward_return > threshold)

        if calibration is not None:
            calibration.record(item.raw_prediction, realized_label)

        store.log_outcome(
            OutcomeLog(
                prediction_id=item.prediction_id,
                resolve_timestamp=str(resolve_ts),
                forward_return=forward_return,
                realized_label=realized_label,
                exit_price=exit_price,
                context=ctx,
            )
        )

    return still_pending


def log_immediate_outcomes(
    store: PredictionStore,
    *,
    timestamp: str,
    symbol: str,
    model_version: str,
    horizon: int,
    p_up: float,
    confidence: float,
    signal: int,
    entry_price: float,
    features: dict[str, float],
    forward_return: float,
    realized_label: int,
    resolve_timestamp: str,
    exit_price: float,
    context: dict[str, Any],
) -> None:
    """Log one horizon prediction + realised outcome (walk-forward backtest path)."""
    from epoch_ai.logging_system.schemas import PredictionLog

    pred_id = store.log_prediction(
        PredictionLog(
            timestamp=timestamp,
            symbol=symbol,
            model_version=model_version,
            horizon=horizon,
            prediction=p_up,
            confidence=confidence,
            signal=signal,
            entry_price=entry_price,
            features=features,
        )
    )
    store.log_outcome(
        OutcomeLog(
            prediction_id=pred_id,
            resolve_timestamp=resolve_timestamp,
            forward_return=forward_return,
            realized_label=realized_label,
            exit_price=exit_price,
            context=context,
        )
    )

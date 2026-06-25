"""Dataclasses describing prediction and outcome log records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class PredictionLog:
    """A single prediction emitted by the model at ``timestamp`` (entry time).

    Attributes:
        timestamp: ISO entry time of the bar the prediction was made on.
        symbol: Trading pair.
        model_version: Identifier of the model that produced the prediction.
        horizon: Forward horizon (in candles) the prediction refers to.
        prediction: Raw model output (P(up) for classification, return otherwise).
        confidence: Distance-from-uncertainty score in ``[0, 1]``.
        signal: Discrete trade signal (1 long, -1 short, 0 flat).
        features: Full feature vector captured at prediction time.
        entry_price: Close price at entry (for outcome computation).
    """

    timestamp: str
    symbol: str
    model_version: str
    horizon: int
    prediction: float
    confidence: float
    signal: int
    features: dict[str, float] = field(default_factory=dict)
    entry_price: float | None = None


@dataclass(slots=True)
class OutcomeLog:
    """The realised outcome of a prediction after its horizon elapses.

    Attributes:
        prediction_id: Foreign key into the predictions table.
        resolve_timestamp: ISO time at which the horizon completed.
        forward_return: Realised return over the horizon.
        realized_label: Binary realised direction (1 up, 0 down).
        exit_price: Close price at horizon end.
        context: Rich influencing context captured during the period (funding shift,
            liquidation spikes, volume spikes, realised volatility, max favourable /
            adverse excursion, etc.).
    """

    prediction_id: int
    resolve_timestamp: str
    forward_return: float
    realized_label: int
    exit_price: float | None = None
    context: dict[str, Any] = field(default_factory=dict)

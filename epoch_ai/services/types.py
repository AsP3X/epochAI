"""Shared runtime datatypes (avoids circular imports between services and execution)."""

from __future__ import annotations

from dataclasses import dataclass

from epoch_ai.execution.risk import RiskDecision


@dataclass(slots=True)
class PredictionResult:
    """A single-bar model output plus risk-adjusted decision."""

    timestamp: str
    raw_prediction: float
    decision: RiskDecision
    model_version: str


@dataclass(slots=True)
class RuntimeStatus:
    """Snapshot for dashboards, bots, and health checks."""

    symbol: str
    timeframe: str
    model_version: str | None
    models_available: int
    task: str

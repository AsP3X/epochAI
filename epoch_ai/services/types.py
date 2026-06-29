"""Shared runtime datatypes (avoids circular imports between services and execution)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from epoch_ai.execution.risk import RiskDecision

# Human: At inference time we lack rolling coverage; use a moderate confidence floor
#        until live history accumulates (Phase 5 wires a config knob).
# Agent: DEFAULT reliability gate for ``HorizonForecast.reliable`` at predict time.
DEFAULT_RELIABILITY_FLOOR = 0.35


@dataclass(slots=True)
class PredictionResult:
    """A single-bar model output plus risk-adjusted decision."""

    timestamp: str
    raw_prediction: float
    decision: RiskDecision
    model_version: str
    features: dict[str, float] | None = None


@dataclass(slots=True)
class RuntimeStatus:
    """Snapshot for dashboards, bots, and health checks."""

    symbol: str
    timeframe: str
    model_version: str | None
    models_available: int
    task: str


@dataclass(slots=True)
class HorizonForecast:
    """One horizon slice of a multi-horizon prediction at bar ``as_of``."""

    label: str
    horizon: int
    target_time: str
    p_up: float
    exp_return: float
    price_p10: float
    price_p50: float
    price_p90: float
    confidence: float
    reliable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "horizon": self.horizon,
            "target_time": self.target_time,
            "p_up": self.p_up,
            "exp_return": self.exp_return,
            "price_p10": self.price_p10,
            "price_p50": self.price_p50,
            "price_p90": self.price_p90,
            "confidence": self.confidence,
            "reliable": self.reliable,
        }


@dataclass(slots=True)
class MultiHorizonPredictionResult:
    """Structured multi-horizon forecast for chart, policy, and JSON APIs."""

    as_of: str
    last_close: float
    model_version: str
    symbol: str
    timeframe: str
    horizons: list[HorizonForecast] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        """Canonical JSON payload for chart + policy consumers."""
        return {
            "as_of": self.as_of,
            "last_close": self.last_close,
            "model_version": self.model_version,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "horizons": [h.to_dict() for h in self.horizons],
        }

    def primary(self) -> HorizonForecast | None:
        """Return the longest-horizon forecast (typically the configured primary head)."""
        if not self.horizons:
            return None
        return max(self.horizons, key=lambda h: h.horizon)


def horizon_confidence(
    p_up: float,
    q10_return: float,
    q90_return: float,
    *,
    band_scale: float = 50.0,
) -> float:
    """Map direction + quantile band width to a confidence score in ``[0, 1]``."""
    decisiveness = min(1.0, abs(p_up - 0.5) * 2.0)
    band = max(1e-8, float(q90_return) - float(q10_return))
    inverse_band = 1.0 / (1.0 + band * band_scale)
    return float(max(0.0, min(1.0, decisiveness * inverse_band)))


def build_multi_horizon_from_structured(
    structured: dict[int, dict[str, Any]],
    row_idx: int,
    *,
    as_of: pd.Timestamp,
    last_close: float,
    model_version: str,
    symbol: str,
    timeframe: str,
    horizons: list[int],
    horizon_label_fn,
    bar_minutes: int,
) -> MultiHorizonPredictionResult:
    """Build a :class:`MultiHorizonPredictionResult` for one row of structured logits."""
    forecasts: list[HorizonForecast] = []
    for h in horizons:
        block = structured[h]
        p_up = block["p_up"]
        q10 = block["q10"]
        q50 = block["q50"]
        q90 = block["q90"]
        if hasattr(p_up, "__len__"):
            p_up = float(p_up[row_idx])
            q10 = float(q10[row_idx])
            q50 = float(q50[row_idx])
            q90 = float(q90[row_idx])
        forecasts.append(
            build_horizon_forecast(
                as_of=as_of,
                last_close=last_close,
                horizon=h,
                horizon_label=horizon_label_fn(h),
                bar_minutes=bar_minutes,
                p_up=float(p_up),
                q10=float(q10),
                q50=float(q50),
                q90=float(q90),
            )
        )
    return MultiHorizonPredictionResult(
        as_of=str(as_of),
        last_close=last_close,
        model_version=model_version,
        symbol=symbol,
        timeframe=timeframe,
        horizons=forecasts,
    )


def build_horizon_forecast(
    *,
    as_of: pd.Timestamp,
    last_close: float,
    horizon: int,
    horizon_label: str,
    bar_minutes: int,
    p_up: float,
    q10: float,
    q50: float,
    q90: float,
    reliability_floor: float = DEFAULT_RELIABILITY_FLOOR,
) -> HorizonForecast:
    """Build one :class:`HorizonForecast` from log-return quantiles and P(up)."""
    target = as_of + pd.Timedelta(minutes=horizon * bar_minutes)
    confidence = horizon_confidence(p_up, q10, q90)
    return HorizonForecast(
        label=horizon_label,
        horizon=horizon,
        target_time=target.isoformat(),
        p_up=float(p_up),
        exp_return=float(q50),
        price_p10=float(last_close * math.exp(q10)),
        price_p50=float(last_close * math.exp(q50)),
        price_p90=float(last_close * math.exp(q90)),
        confidence=confidence,
        reliable=confidence >= reliability_floor,
    )

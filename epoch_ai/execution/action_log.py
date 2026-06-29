"""Structured action/outcome log for the feedback-loop retrain path."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from epoch_ai.execution.risk import RiskDecision
from epoch_ai.services.types import MultiHorizonPredictionResult


@dataclass(slots=True)
class ActionRecord:
    """One replayable (observation, prediction, decision, fill) row."""

    timestamp: str
    symbol: str
    model_version: str
    policy_backend: str
    raw_prediction: float
    decision_signal: int
    decision_weight: float
    equity: float
    position_weight: float
    forecast: dict[str, Any] | None = None
    fill_fee: float | None = None
    bar_return: float | None = None


class ActionLog:
    """Append-only JSONL store for bot experience."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: ActionRecord) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(record), default=float))
            fh.write("\n")

    def log_step(
        self,
        *,
        timestamp: str,
        symbol: str,
        model_version: str,
        policy_backend: str,
        raw_prediction: float,
        decision: RiskDecision,
        equity: float,
        position_weight: float,
        multi: MultiHorizonPredictionResult | None = None,
        fill_fee: float | None = None,
        bar_return: float | None = None,
    ) -> None:
        self.append(
            ActionRecord(
                timestamp=timestamp,
                symbol=symbol,
                model_version=model_version,
                policy_backend=policy_backend,
                raw_prediction=raw_prediction,
                decision_signal=decision.signal,
                decision_weight=decision.target_weight,
                equity=equity,
                position_weight=position_weight,
                forecast=multi.to_json() if multi is not None else None,
                fill_fee=fill_fee,
                bar_return=bar_return,
            )
        )


def load_records(path: str | Path) -> list[ActionRecord]:
    """Load all JSONL rows from the action log (empty when missing)."""
    path = Path(path)
    if not path.exists():
        return []
    records: list[ActionRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        records.append(ActionRecord(**data))
    return records


def load_frame(path: str | Path) -> pd.DataFrame:
    """Load the action log as a DataFrame for retrain weighting."""
    records = load_records(path)
    if not records:
        return pd.DataFrame()
    return pd.DataFrame([asdict(r) for r in records])


def boost_weights_from_action_log(
    weights: np.ndarray,
    timestamps: pd.Index | pd.Series | None,
    action_df: pd.DataFrame,
    boost: float,
) -> np.ndarray:
    """Up-weight rows whose timestamps appear in the live action log.

    Returns ``weights`` unchanged when there is nothing to boost, including the
    ``weights is None`` case (recency weighting disabled), so callers can pass the
    raw output of :func:`recency_weights` without a separate guard.
    """
    if weights is None or timestamps is None or action_df.empty or boost <= 1.0:
        return weights
    if "timestamp" not in action_df.columns:
        return weights
    logged = set(action_df["timestamp"].astype(str))
    out = weights.copy()
    for i, ts in enumerate(timestamps):
        if str(ts) in logged:
            out[i] *= boost
    return out

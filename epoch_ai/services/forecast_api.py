"""JSON payloads for live chart cones and historical predicted-vs-realized overlays."""

from __future__ import annotations

from typing import Any

import pandas as pd

from epoch_ai.execution.policy.baseline import baseline_policy
from epoch_ai.logging_system.store import PredictionStore
from epoch_ai.services.types import MultiHorizonPredictionResult


def build_live_payload(result: MultiHorizonPredictionResult) -> dict[str, Any]:
    """Forward-looking cone payload for live chart consumers."""
    baseline = baseline_policy(result.horizons)
    payload = result.to_json()
    payload["type"] = "live"
    payload["baseline"] = {
        "signal": baseline.signal,
        "confidence": baseline.confidence,
        "weighted_p_up": baseline.weighted_p_up,
        "n_heads_used": baseline.n_heads_used,
        "skipped_horizons": list(baseline.skipped_horizons),
    }
    return payload


def build_historical_payload(
    store: PredictionStore,
    *,
    symbol: str,
    limit: int = 500,
) -> dict[str, Any]:
    """Historical predicted-vs-realized series from the SQLite log."""
    preds = store.predictions_frame(symbol=symbol)
    if preds.empty:
        return {"type": "historical", "symbol": symbol, "series": []}

    outcomes = store.outcomes_frame()
    if not outcomes.empty:
        joined = preds.merge(outcomes, left_on="id", right_on="prediction_id", how="left")
    else:
        joined = preds.copy()
        joined["forward_return"] = float("nan")
        joined["realized_label"] = float("nan")

    joined = joined.sort_values("timestamp").tail(limit)
    series: list[dict[str, Any]] = []
    for row in joined.itertuples(index=False):
        features = {}
        raw_features = getattr(row, "features", None)
        if raw_features:
            import json

            try:
                features = json.loads(raw_features)
            except (TypeError, json.JSONDecodeError):
                features = {}
        series.append(
            {
                "timestamp": row.timestamp,
                "horizon": int(row.horizon),
                "p_up": float(row.prediction),
                "confidence": float(row.confidence),
                "signal": int(row.signal),
                "forward_return": (
                    float(row.forward_return)
                    if getattr(row, "forward_return", None) is not None
                    and not pd.isna(row.forward_return)
                    else None
                ),
                "realized_label": (
                    int(row.realized_label)
                    if getattr(row, "realized_label", None) is not None
                    and not pd.isna(row.realized_label)
                    else None
                ),
                "bands": {
                    k: features[k]
                    for k in ("return_q10", "return_q50", "return_q90", "price_p50")
                    if k in features
                },
            }
        )
    return {"type": "historical", "symbol": symbol, "series": series}

"""Tests for multi-horizon forecast API payloads."""

from __future__ import annotations

import json

import pytest

from epoch_ai.execution.policy.baseline import baseline_policy
from epoch_ai.logging_system.schemas import OutcomeLog, PredictionLog
from epoch_ai.logging_system.store import PredictionStore
from epoch_ai.services.forecast_api import build_historical_payload, build_live_payload
from epoch_ai.services.types import HorizonForecast, MultiHorizonPredictionResult

pytestmark = pytest.mark.slow


def _sample_result() -> MultiHorizonPredictionResult:
    forecasts = [
        HorizonForecast(
            label="5m",
            horizon=5,
            target_time="2020-01-01T00:05:00",
            p_up=0.62,
            exp_return=0.001,
            price_p10=99.0,
            price_p50=100.1,
            price_p90=101.0,
            confidence=0.55,
            reliable=True,
        ),
        HorizonForecast(
            label="1hr",
            horizon=60,
            target_time="2020-01-01T01:00:00",
            p_up=0.48,
            exp_return=-0.0005,
            price_p10=98.0,
            price_p50=99.5,
            price_p90=101.5,
            confidence=0.12,
            reliable=False,
        ),
    ]
    return MultiHorizonPredictionResult(
        as_of="2020-01-01T00:00:00",
        last_close=100.0,
        model_version="v_test",
        symbol="BTC/USDT",
        timeframe="1m",
        horizons=forecasts,
    )


def test_live_payload_includes_baseline():
    payload = build_live_payload(_sample_result())
    assert payload["type"] == "live"
    assert payload["model_version"] == "v_test"
    assert len(payload["horizons"]) == 2
    assert "baseline" in payload
    assert payload["baseline"]["n_heads_used"] >= 1


def test_baseline_skips_unreliable_heads():
    decision = baseline_policy(_sample_result().horizons)
    assert decision.n_heads_used == 1
    assert 60 in decision.skipped_horizons


def test_historical_payload_from_store(tmp_path):
    db = tmp_path / "pred.sqlite"
    store = PredictionStore(str(db))
    pred_id = store.log_prediction(
        PredictionLog(
            timestamp="2020-01-01T00:00:00",
            symbol="BTC/USDT",
            model_version="v_1",
            horizon=5,
            prediction=0.6,
            confidence=0.5,
            signal=1,
            entry_price=100.0,
            features={"return_q50": 0.001, "price_p50": 100.1},
        )
    )
    store.log_outcome(
        OutcomeLog(
            prediction_id=pred_id,
            resolve_timestamp="2020-01-01T00:05:00",
            forward_return=0.002,
            realized_label=1,
            exit_price=100.2,
        )
    )
    payload = build_historical_payload(store, symbol="BTC/USDT")
    store.close()
    assert payload["type"] == "historical"
    assert len(payload["series"]) == 1
    assert payload["series"][0]["forward_return"] == pytest.approx(0.002)


def test_multi_horizon_result_to_json_roundtrip():
    raw = _sample_result().to_json()
    parsed = json.loads(json.dumps(raw))
    assert parsed["horizons"][0]["label"] == "5m"

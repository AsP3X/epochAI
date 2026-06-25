"""Tests for the SQLite prediction/outcome store and joiner."""

from __future__ import annotations

from epoch_ai.logging_system.joiner import build_training_dataset
from epoch_ai.logging_system.schemas import OutcomeLog, PredictionLog
from epoch_ai.logging_system.store import PredictionStore


def test_log_and_join(tmp_path):
    store = PredictionStore(str(tmp_path / "p.sqlite"))
    pred_id = store.log_prediction(
        PredictionLog(
            timestamp="2020-01-01T00:00:00Z",
            symbol="BTC/USDT",
            model_version="v_1",
            horizon=8,
            prediction=0.7,
            confidence=0.4,
            signal=1,
            entry_price=100.0,
            features={"ta_rsi_14": 55.0, "vol_std_24": 0.01},
        )
    )
    store.log_outcome(
        OutcomeLog(
            prediction_id=pred_id,
            resolve_timestamp="2020-01-01T02:00:00Z",
            forward_return=0.03,
            realized_label=1,
            exit_price=103.0,
            context={"volume_spike": 2.5, "funding_shift": 0.0001},
        )
    )

    counts = store.counts()
    assert counts == {"predictions": 1, "outcomes": 1}

    dataset = build_training_dataset(store, "BTC/USDT")
    assert len(dataset) == 1
    assert dataset["target"].iloc[0] == 1
    assert "ta_rsi_14" in dataset.columns
    store.close()

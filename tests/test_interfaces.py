"""Tests for web/Telegram adapters and multi-horizon logging helpers."""

from __future__ import annotations

import pandas as pd

from epoch_ai.interfaces.telegram import format_forecast_summary, format_trade_alert
from epoch_ai.interfaces.web import build_dashboard_payload
from epoch_ai.logging_system.multi_horizon_log import (
    PendingHorizonLog,
    resolve_pending_horizons,
)
from epoch_ai.logging_system.schemas import PredictionLog
from epoch_ai.logging_system.store import PredictionStore
from epoch_ai.services.forecast_api import build_live_payload
from epoch_ai.services.runtime import RuntimeService
from epoch_ai.services.types import HorizonForecast, MultiHorizonPredictionResult


def test_telegram_forecast_summary():
    payload = build_live_payload(
        MultiHorizonPredictionResult(
            as_of="2020-01-01T00:00:00",
            last_close=100.0,
            model_version="v_1",
            symbol="BTC/USDT",
            timeframe="1m",
            horizons=[
                HorizonForecast(
                    label="5m",
                    horizon=5,
                    target_time="2020-01-01T00:05:00",
                    p_up=0.61,
                    exp_return=0.001,
                    price_p10=99.0,
                    price_p50=100.1,
                    price_p90=101.0,
                    confidence=0.55,
                    reliable=True,
                )
            ],
        )
    )
    text = format_forecast_summary(payload)
    assert "5m" in text
    assert "baseline" in text.lower() or "signal=" in text


def test_trade_alert_format():
    msg = format_trade_alert(
        symbol="BTC/USDT",
        timestamp="2020-01-01",
        signal=1,
        price=100.0,
        equity=10100.0,
        model_version="v_1",
    )
    assert "LONG" in msg


def test_resolve_pending_horizons(tmp_path):
    db = tmp_path / "pred.sqlite"
    store = PredictionStore(str(db))
    idx = pd.date_range("2020-01-01", periods=20, freq="15min")
    close = pd.Series([100 + i * 0.1 for i in range(20)], index=idx)
    pred_id = store.log_prediction(
        PredictionLog(
            timestamp=str(idx[0]),
            symbol="BTC/USDT",
            model_version="v_1",
            horizon=2,
            prediction=0.6,
            confidence=0.5,
            signal=1,
            entry_price=100.0,
            features={},
        )
    )
    pending = [
        PendingHorizonLog(
            prediction_id=pred_id,
            entry_index=0,
            entry_price=100.0,
            horizon=2,
            raw_prediction=0.6,
        )
    ]
    still = resolve_pending_horizons(
        pending,
        current_index=2,
        close=close,
        index=idx,
        threshold=0.0,
        store=store,
    )
    assert still == []
    outcomes = store.outcomes_frame()
    store.close()
    assert len(outcomes) == 1


def test_dashboard_payload_without_model(small_config, market, tmp_path, monkeypatch):
    from epoch_ai.data.downloader import HistoricalDownloader

    small_config.model.model_dir = str(tmp_path / "models")
    small_config.logging.db_path = str(tmp_path / "logs.sqlite")

    def fake_load(self, symbol=None, *, n_bars=None, force=False, skip_enrichment=False):
        cap = len(market) if n_bars is None else min(n_bars, len(market))
        return market.iloc[:cap].copy()

    monkeypatch.setattr(HistoricalDownloader, "load_or_download", fake_load)

    runtime = RuntimeService(small_config)
    with PredictionStore(small_config.logging.db_path) as store:
        payload = build_dashboard_payload(small_config, runtime, store=store, n_bars=500)
    assert "live" in payload
    assert payload["live"]["error"] == "no_trained_model"

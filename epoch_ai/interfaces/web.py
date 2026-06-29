"""Web/dashboard JSON adapter (no charting logic in the engine)."""

from __future__ import annotations

from typing import Any

from epoch_ai.config.settings import AppConfig
from epoch_ai.data.downloader import HistoricalDownloader
from epoch_ai.execution.kill_switch import KillSwitch
from epoch_ai.logging_system.store import PredictionStore
from epoch_ai.services.forecast_api import build_historical_payload, build_live_payload
from epoch_ai.services.runtime import RuntimeService


def build_dashboard_payload(
    config: AppConfig,
    runtime: RuntimeService,
    *,
    store: PredictionStore | None = None,
    n_bars: int | None = None,
    historical_limit: int = 500,
) -> dict[str, Any]:
    """Assemble status + live cone + historical overlay for a web client."""
    market = HistoricalDownloader(config).load_or_download(
        config.primary_symbol,
        n_bars=n_bars,
    )
    if runtime.status().models_available == 0:
        live_payload: dict[str, Any] = {
            "type": "live",
            "error": "no_trained_model",
            "horizons": [],
        }
    else:
        runtime.load_model()
        live_payload = build_live_payload(runtime.predict_multi_horizon(market))

    historical_payload: dict[str, Any] = {"type": "historical", "symbol": config.primary_symbol, "series": []}
    if store is not None:
        historical_payload = build_historical_payload(
            store,
            symbol=config.primary_symbol,
            limit=historical_limit,
        )

    st = runtime.status()
    ks = KillSwitch(config.execution.kill_switch_path).read()
    return {
        "status": {
            "symbol": st.symbol,
            "timeframe": st.timeframe,
            "model_version": st.model_version,
            "models_available": st.models_available,
            "task": st.task,
            "kill_switch": {"halted": ks.halted, "reason": ks.reason},
        },
        "live": live_payload,
        "historical": historical_payload,
    }

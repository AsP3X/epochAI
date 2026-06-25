"""FastAPI HTTP surface for train/run/monitor operations."""

from __future__ import annotations

from typing import Any

from epoch_ai.config.settings import AppConfig
from epoch_ai.execution.kill_switch import KillSwitch
from epoch_ai.models.registry import ModelRegistry
from epoch_ai.monitoring.health import check_live_health
from epoch_ai.services.runtime import RuntimeService
from epoch_ai.services.training import TrainingService


def create_app(config: AppConfig):
    """Build a FastAPI application bound to ``config``."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
    except ImportError as exc:
        raise RuntimeError(
            "FastAPI is required for the HTTP API. "
            "pip install -r requirements-optional.txt"
        ) from exc

    app = FastAPI(title="epochAI", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    runtime = RuntimeService(config)
    kill_switch = KillSwitch(config.execution.kill_switch_path)

    class HaltRequest(BaseModel):
        reason: str = "manual halt via API"

    class TrainRequest(BaseModel):
        bars: int | None = None
        max_steps: int | None = None
        log_predictions: bool = False

    class ExportRequest(BaseModel):
        dest: str = "artifacts/exports"
        model_version: str | None = None

    @app.get("/health")
    def health() -> dict[str, Any]:
        live = check_live_health(config)
        return {
            "status": "ok" if live.ready else "degraded",
            "issues": live.issues,
        }

    @app.get("/status")
    def status() -> dict[str, Any]:
        st = runtime.status()
        ks = kill_switch.read()
        return {
            "symbol": st.symbol,
            "timeframe": st.timeframe,
            "model_version": st.model_version,
            "models_available": st.models_available,
            "task": st.task,
            "kill_switch": {"halted": ks.halted, "reason": ks.reason},
        }

    @app.get("/models")
    def models() -> list[dict[str, Any]]:
        return ModelRegistry(config.model.model_dir).list_versions()

    @app.post("/predict/latest")
    def predict_latest(bars: int = 1200) -> dict[str, Any]:
        from epoch_ai.data.downloader import HistoricalDownloader

        market = HistoricalDownloader(config).load_or_download(
            config.primary_symbol,
            n_bars=bars,
        )
        if runtime.status().models_available == 0:
            raise HTTPException(status_code=400, detail="No trained models available.")
        runtime.load_model()
        result = runtime.predict_market(market)
        return {
            "timestamp": result.timestamp,
            "raw_prediction": result.raw_prediction,
            "signal": result.decision.signal,
            "confidence": result.decision.confidence,
            "model_version": result.model_version,
        }

    @app.post("/train")
    def train(body: TrainRequest) -> dict[str, Any]:
        service = TrainingService(config)
        result = service.train(
            n_bars=body.bars,
            max_steps=body.max_steps,
            log_predictions=body.log_predictions,
        )
        return {
            "model_version": result.model_version,
            "walk_forward_steps": result.walk_forward_steps,
            "train_rows": result.train_rows,
        }

    @app.post("/export")
    def export_model(body: ExportRequest) -> dict[str, str]:
        from epoch_ai.export.model_card import export_bundle_with_card

        path = export_bundle_with_card(
            config,
            dest=body.dest,
            label=body.model_version,
        )
        return {"bundle_path": str(path)}

    @app.post("/kill/halt")
    def halt(body: HaltRequest) -> dict[str, Any]:
        state = kill_switch.halt(body.reason)
        return {"halted": state.halted, "reason": state.reason, "updated_at": state.updated_at}

    @app.post("/kill/resume")
    def resume() -> dict[str, Any]:
        state = kill_switch.resume()
        return {"halted": state.halted, "updated_at": state.updated_at}

    @app.get("/kill")
    def kill_status() -> dict[str, Any]:
        state = kill_switch.read()
        return {"halted": state.halted, "reason": state.reason, "updated_at": state.updated_at}

    return app

"""Runtime mode — load a trained model and produce predictions / paper execution.

This module is the programmatic entry point for **running the AI** after training.
Future Telegram and website interfaces should call :class:`RuntimeService` for
predictions, status, and paper/live sessions.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd

from epoch_ai.config.settings import AppConfig
from epoch_ai.data.downloader import HistoricalDownloader
from epoch_ai.execution.live_engine import LiveSessionResult, LiveTradingEngine
from epoch_ai.execution.live_loop import LiveLoopResult, run_bar_loop
from epoch_ai.execution.risk import RiskManager
from epoch_ai.features.pipeline import FeaturePipeline, build_target, forward_return
from epoch_ai.models.base import BaseModel
from epoch_ai.models.registry import ModelRegistry
from epoch_ai.services.types import PredictionResult, RuntimeStatus
from epoch_ai.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["PredictionResult", "RuntimeService", "RuntimeStatus"]


class RuntimeService:
    """Load trained models and run inference + paper execution."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.registry = ModelRegistry(config.model.model_dir)
        self.pipeline = FeaturePipeline(config)
        self.risk = RiskManager(config.risk, config.prediction)
        self._model: BaseModel | None = None
        self._model_version: str | None = None
        self._metadata: dict[str, Any] = {}

    def status(self) -> RuntimeStatus:
        """Return runtime readiness (model registry + config summary)."""
        versions = self.registry.list_versions()
        latest = versions[-1]["label"] if versions else None
        return RuntimeStatus(
            symbol=self.config.primary_symbol,
            timeframe=self.config.timeframe,
            model_version=self._model_version or latest,
            models_available=len(versions),
            task=self.config.prediction.task,
        )

    def load_model(self, version: str | None = None) -> str:
        """Load a model from the registry; return its version label."""
        model, meta = self.registry.load(
            version,
            self.config.model,
            task=self.config.prediction.task,
        )
        self._model = model
        self._model_version = meta["label"]
        self._metadata = meta
        return self._model_version

    def _require_model(self) -> BaseModel:
        if self._model is None:
            self.load_model()
        assert self._model is not None
        return self._model

    def predict_market(self, market: pd.DataFrame) -> PredictionResult:
        """Predict on the **latest** bar of a OHLCV frame."""
        if market.empty:
            raise ValueError("Cannot predict on empty market data.")
        # Human: Live ticks call this every bar; skip noisy pipeline INFO each time.
        # Agent: log_stats=False; RETURNS feature dict for SQLite logging without re-transform.
        features = self.pipeline.transform(market, log_stats=False)
        if features.empty:
            raise ValueError(
                f"Feature pipeline produced no rows for the {len(market)}-bar window "
                "(all rows dropped as feature warm-up/NaN). Provide a longer warmup "
                "window or data with the required context columns (e.g. funding_rate)."
            )
        model = self._require_model()
        ts = market.index[-1]
        row = features.iloc[[-1]]
        raw = float(model.predict(row)[0])
        decision = self.risk.decide(raw)
        return PredictionResult(
            timestamp=str(ts),
            raw_prediction=raw,
            decision=decision,
            model_version=self._model_version or "unknown",
            features={k: float(v) for k, v in row.iloc[0].items()},
        )

    def run_session(
        self,
        *,
        mode: Literal["paper", "replay"] = "paper",
        n_bars: int | None = None,
        live_bars: int = 500,
        retrain_every: int = 0,
        model_version: str | None = None,
    ) -> LiveLoopResult:
        """Execute a bar-by-bar paper session using a registered model.

        Args:
            mode: ``paper`` or ``replay`` (both use historical replay today; live
                WebSocket remains on the ``live`` CLI without registry model).
            n_bars: Historical depth to load.
            live_bars: Tail length to simulate.
            retrain_every: Inline retrain cadence (0 = use frozen registry model).
            model_version: Registry label to load; latest when ``None``.

        Raises:
            FileNotFoundError: When no model exists in the registry.
        """
        del mode  # reserved for future live-stream distinction
        if model_version is not None or retrain_every == 0:
            model, meta = self.registry.load(
                model_version,
                self.config.model,
                task=self.config.prediction.task,
            )
            self._model = model
            self._model_version = meta["label"]
        else:
            model = None

        market = HistoricalDownloader(self.config).load_or_download(
            self.config.primary_symbol,
            n_bars=n_bars,
        )
        features = self.pipeline.transform(market)
        y = build_target(market, self.config.prediction)
        fwd = forward_return(market, self.config.prediction.horizon)
        data = features.join(y).join(fwd).dropna(subset=["target", "forward_return"])

        live_bars = min(live_bars, len(data) - self.config.walk_forward.initial_train_period)
        if live_bars < 1:
            raise ValueError("Not enough data for runtime session; increase n_bars.")

        split = len(data) - live_bars
        return run_bar_loop(
            self.config,
            market,
            start_pos=split,
            retrain_every=retrain_every,
            model=model,
        )

    def run_live_feed(
        self,
        *,
        n_bars: int | None = None,
        feed_bars: int | None = None,
        model_version: str | None = None,
        log_predictions: bool = False,
        warmup_bars: int | None = None,
    ) -> LiveSessionResult:
        """Simulate a live feed by growing an OHLCV buffer bar-by-bar, then trading.

        Uses historical/synthetic data as the data source (offline-safe). Real
        WebSocket streaming uses the same :class:`LiveTradingEngine` via ``run_live_stream``.
        """
        market = HistoricalDownloader(self.config).load_or_download(
            self.config.primary_symbol,
            n_bars=n_bars,
        )
        engine = LiveTradingEngine.create(
            self.config,
            model_version=model_version,
            log_predictions=log_predictions,
        )
        min_bars = warmup_bars or engine.min_buffer_bars
        if len(market) <= min_bars:
            raise ValueError(
                f"Need more than {min_bars} bars for live warmup; got {len(market)}."
            )

        end = len(market) if feed_bars is None else min(len(market), min_bars + feed_bars)
        symbol = self.config.primary_symbol
        for i in range(min_bars, end):
            window = market.iloc[: i + 1]
            engine.process_bar(symbol, window)
        return engine.finish()

    async def run_live_stream(
        self,
        *,
        model_version: str | None = None,
        log_predictions: bool = False,
        warmup_bars: int | None = None,
    ) -> LiveSessionResult:
        """Stream live candles from the exchange and trade on each new bar."""
        import asyncio

        from epoch_ai.data.websocket import RealtimeDataHandler

        market = HistoricalDownloader(self.config).load_or_download(
            self.config.primary_symbol,
            n_bars=warmup_bars or self.config.execution.min_buffer_bars,
        )
        engine = LiveTradingEngine.create(
            self.config,
            model_version=model_version,
            log_predictions=log_predictions,
        )
        handler = RealtimeDataHandler(self.config)
        symbol = self.config.primary_symbol

        # Seed WebSocket buffer with warmup history.
        for idx, ts in enumerate(market.index):
            row = market.iloc[idx]
            ts_ms = int(pd.Timestamp(ts).timestamp() * 1000)
            handler.ingest_candle(
                symbol,
                [
                    ts_ms,
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    float(row.get("volume", 0.0)),
                ],
            )

        def on_candle(sym: str, frame: pd.DataFrame) -> None:
            tick = engine.process_bar(sym, frame)
            if tick is not None:
                logger.info(
                    "Live tick %s pred=%.3f signal=%d equity=%.2f",
                    tick.prediction.timestamp,
                    tick.prediction.raw_prediction,
                    tick.prediction.decision.signal,
                    tick.equity,
                )

        try:
            await handler.stream(on_candle=on_candle)
        except asyncio.CancelledError:
            pass
        return engine.finish()

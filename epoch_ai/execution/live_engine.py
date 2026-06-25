"""Live trading engine: live data -> predict -> execute -> log outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from epoch_ai.calibration.tracker import CalibrationTracker
from epoch_ai.config.settings import AppConfig
from epoch_ai.execution.audit_log import AuditLog
from epoch_ai.execution.executor import Fill, TradeExecutor, build_executor
from epoch_ai.execution.kill_switch import KillSwitch
from epoch_ai.execution.portfolio_state import PortfolioState
from epoch_ai.execution.treasury import Treasury, TreasurySnapshot
from epoch_ai.features.pipeline import FeaturePipeline
from epoch_ai.logging_system.schemas import OutcomeLog, PredictionLog
from epoch_ai.logging_system.store import PredictionStore
from epoch_ai.monitoring.metrics import MetricsRecorder
from epoch_ai.services.types import PredictionResult
from epoch_ai.utils.logging import get_logger

if TYPE_CHECKING:
    from epoch_ai.services.runtime import RuntimeService

logger = get_logger(__name__)


@dataclass(slots=True)
class PendingPrediction:
    """Prediction awaiting horizon resolution for outcome logging."""

    prediction_id: int
    entry_index: int
    entry_price: float
    raw_prediction: float


@dataclass(slots=True)
class LiveTickResult:
    """Result of processing one new live bar."""

    prediction: PredictionResult
    fill: Fill | None
    equity: float
    trading_capital: float
    reserved_wins: float
    halted: bool = False
    calibration_blocked: bool = False


@dataclass(slots=True)
class LiveSessionResult:
    """Summary when a live feed session ends."""

    ticks: int
    fills: int
    final_equity: float
    treasury: TreasurySnapshot
    model_version: str
    calibration_gate_passed: bool = True


class LiveTradingEngine:
    """Connect live OHLCV feeds to predictions and trade execution."""

    def __init__(
        self,
        config: AppConfig,
        runtime: RuntimeService,
        executor: TradeExecutor,
        treasury: Treasury,
        store: PredictionStore | None = None,
        *,
        kill_switch: KillSwitch | None = None,
        audit_log: AuditLog | None = None,
        calibration: CalibrationTracker | None = None,
        metrics: MetricsRecorder | None = None,
    ) -> None:
        self.config = config
        self.runtime = runtime
        self.executor = executor
        self.treasury = treasury
        self.store = store
        self.kill_switch = kill_switch or KillSwitch(config.execution.kill_switch_path)
        self.audit_log = audit_log
        self.calibration = calibration
        self.metrics = metrics
        self._pending: list[PendingPrediction] = []
        self._portfolio: PortfolioState | None = None
        self._prev_close: float | None = None
        self._tick_count = 0
        self._fill_count = 0
        self._calibration_gate_passed = True

    @classmethod
    def create(
        cls,
        config: AppConfig,
        *,
        model_version: str | None = None,
        log_predictions: bool = False,
    ) -> LiveTradingEngine:
        """Build engine with registry model, treasury, and executor."""
        from epoch_ai.services.runtime import RuntimeService

        exec_cfg = config.execution
        treasury = Treasury.load_or_create(
            initial_capital=config.risk.initial_capital,
            reserve_fraction=exec_cfg.reserve_fraction,
            cold_storage_fraction=exec_cfg.cold_storage_fraction,
            max_daily_profit_take=exec_cfg.max_daily_profit_take,
            state_path=exec_cfg.treasury_state_path,
        )
        runtime = RuntimeService(config)
        runtime.load_model(model_version)
        executor = build_executor(config, treasury)
        store = PredictionStore(config.logging.db_path) if log_predictions else None
        audit_log = AuditLog(exec_cfg.audit_log_path) if exec_cfg.audit_enabled else None
        metrics = MetricsRecorder(exec_cfg.metrics_path) if exec_cfg.metrics_enabled else None
        calibration = CalibrationTracker(
            min_accuracy=exec_cfg.calibration_min_accuracy,
            min_samples=exec_cfg.calibration_min_samples,
        )
        return cls(
            config,
            runtime,
            executor,
            treasury,
            store,
            kill_switch=KillSwitch(exec_cfg.kill_switch_path),
            audit_log=audit_log,
            calibration=calibration,
            metrics=metrics,
        )

    @property
    def min_buffer_bars(self) -> int:
        return max(
            self.config.execution.min_buffer_bars,
            self.config.walk_forward.initial_train_period,
        )

    def process_bar(self, symbol: str, market: pd.DataFrame) -> LiveTickResult | None:
        """Ingest the latest bar, predict, execute, and log."""
        if len(market) < self.min_buffer_bars:
            logger.debug(
                "Warmup: %d/%d bars for %s",
                len(market),
                self.min_buffer_bars,
                symbol,
            )
            return None

        if self._portfolio is None:
            self._portfolio = PortfolioState.initial(self.executor.equity)

        self._resolve_outcomes(market)
        close = float(market["close"].iloc[-1])
        ts = market.index[-1]

        if self._prev_close is not None and self._prev_close > 0:
            period_return = close / self._prev_close - 1.0
            prev_eq = self.executor.equity
            self.executor.mark_to_market(period_return)
            if self._portfolio is not None:
                lost = self.executor.equity < prev_eq
                self._portfolio.after_bar(
                    self.executor.equity,
                    lost_trade=lost,
                    cooldown_bars=self.config.risk.cooldown_bars,
                )

        pred = self.runtime.predict_market(market)
        if self._portfolio is not None:
            pred.decision = self.runtime.risk.decide(
                pred.raw_prediction,
                self._portfolio,
            )

        features = FeaturePipeline(self.config).transform(market)
        feature_row = features.iloc[-1].to_dict()

        halted = self.kill_switch.is_halted()
        calibration_blocked = False
        fill: Fill | None = None

        if halted:
            logger.warning("Kill switch active — skipping rebalance.")
            if self.audit_log is not None:
                self.audit_log.append(
                    "halt_skip",
                    {
                        "symbol": symbol,
                        "timestamp": str(ts),
                        "reason": self.kill_switch.read().reason,
                    },
                )
        else:
            gate = self.calibration.check_gate() if self.calibration is not None else None
            if gate is not None and not gate.passed:
                calibration_blocked = True
                self._calibration_gate_passed = False
                logger.warning(
                    "Calibration gate failed (acc=%.3f, n=%d) — skipping rebalance.",
                    gate.mean_accuracy,
                    gate.n_samples,
                )
                if self.audit_log is not None:
                    self.audit_log.append(
                        "calibration_block",
                        {
                            "symbol": symbol,
                            "timestamp": str(ts),
                            "mean_accuracy": gate.mean_accuracy,
                            "n_samples": gate.n_samples,
                        },
                    )
            else:
                fill = self.executor.rebalance(str(ts), close, pred.decision)
                if fill is not None:
                    self._fill_count += 1
                    if self.audit_log is not None:
                        self.audit_log.append(
                            "fill",
                            {
                                "symbol": symbol,
                                "timestamp": str(ts),
                                "price": close,
                                "signal": pred.decision.signal,
                                "target_weight": pred.decision.target_weight,
                                "equity": self.executor.equity,
                            },
                        )

        if self.audit_log is not None:
            self.audit_log.append(
                "prediction",
                {
                    "symbol": symbol,
                    "timestamp": str(ts),
                    "model_version": pred.model_version,
                    "raw_prediction": pred.raw_prediction,
                    "signal": pred.decision.signal,
                    "confidence": pred.decision.confidence,
                },
            )

        if self.store is not None:
            pred_id = self.store.log_prediction(
                PredictionLog(
                    timestamp=str(ts),
                    symbol=symbol,
                    model_version=pred.model_version,
                    horizon=self.config.prediction.horizon,
                    prediction=pred.raw_prediction,
                    confidence=pred.decision.confidence,
                    signal=pred.decision.signal,
                    entry_price=close,
                    features={k: float(v) for k, v in feature_row.items()},
                )
            )
            self._pending.append(
                PendingPrediction(
                    prediction_id=pred_id,
                    entry_index=len(market) - 1,
                    entry_price=close,
                    raw_prediction=pred.raw_prediction,
                )
            )

        if self.metrics is not None:
            self.metrics.record(
                "live_tick",
                {
                    "symbol": symbol,
                    "equity": self.executor.equity,
                    "trading_capital": self.treasury.trading_capital,
                    "raw_prediction": pred.raw_prediction,
                    "signal": pred.decision.signal,
                    "halted": halted,
                    "calibration_blocked": calibration_blocked,
                },
            )

        self._prev_close = close
        self._tick_count += 1
        return LiveTickResult(
            prediction=pred,
            fill=fill,
            equity=self.executor.equity,
            trading_capital=self.treasury.trading_capital,
            reserved_wins=self.treasury.reserved_wins,
            halted=halted,
            calibration_blocked=calibration_blocked,
        )

    def _resolve_outcomes(self, market: pd.DataFrame) -> None:
        """Log realised outcomes once the prediction horizon has elapsed."""
        if not self._pending:
            return
        horizon = self.config.prediction.horizon
        threshold = self.config.prediction.threshold
        current_idx = len(market) - 1
        still_pending: list[PendingPrediction] = []

        for pending in self._pending:
            if current_idx - pending.entry_index < horizon:
                still_pending.append(pending)
                continue
            resolve_idx = min(pending.entry_index + horizon, current_idx)
            exit_price = float(market["close"].iloc[resolve_idx])
            forward_return = exit_price / pending.entry_price - 1.0
            realized_label = int(forward_return > threshold)
            resolve_ts = market.index[resolve_idx]

            if self.calibration is not None:
                self.calibration.record(pending.raw_prediction, realized_label)

            if self.store is not None:
                self.store.log_outcome(
                    OutcomeLog(
                        prediction_id=pending.prediction_id,
                        resolve_timestamp=str(resolve_ts),
                        forward_return=forward_return,
                        realized_label=realized_label,
                        exit_price=exit_price,
                        context={"live": True},
                    )
                )

        self._pending = still_pending

    def finish(self) -> LiveSessionResult:
        """Settle treasury and close resources."""
        if self.store is not None:
            self.store.close()
        snapshot = self.executor.settle_session()
        return LiveSessionResult(
            ticks=self._tick_count,
            fills=self._fill_count,
            final_equity=self.executor.equity,
            treasury=snapshot,
            model_version=self.runtime.status().model_version or "unknown",
            calibration_gate_passed=self._calibration_gate_passed,
        )

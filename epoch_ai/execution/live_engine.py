"""Live trading engine: live data -> predict -> execute -> log outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from epoch_ai.config.settings import AppConfig
from epoch_ai.execution.executor import Fill, TradeExecutor, build_executor
from epoch_ai.execution.portfolio_state import PortfolioState
from epoch_ai.execution.treasury import Treasury, TreasurySnapshot
from epoch_ai.features.pipeline import FeaturePipeline
from epoch_ai.logging_system.schemas import OutcomeLog, PredictionLog
from epoch_ai.logging_system.store import PredictionStore
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


@dataclass(slots=True)
class LiveTickResult:
    """Result of processing one new live bar."""

    prediction: PredictionResult
    fill: Fill | None
    equity: float
    trading_capital: float
    reserved_wins: float


@dataclass(slots=True)
class LiveSessionResult:
    """Summary when a live feed session ends."""

    ticks: int
    fills: int
    final_equity: float
    treasury: TreasurySnapshot
    model_version: str


class LiveTradingEngine:
    """Connect live OHLCV feeds to predictions and trade execution."""

    def __init__(
        self,
        config: AppConfig,
        runtime: RuntimeService,
        executor: TradeExecutor,
        treasury: Treasury,
        store: PredictionStore | None = None,
    ) -> None:
        self.config = config
        self.runtime = runtime
        self.executor = executor
        self.treasury = treasury
        self.store = store
        self._pending: list[PendingPrediction] = []
        self._portfolio: PortfolioState | None = None
        self._prev_close: float | None = None
        self._tick_count = 0

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

        treasury = Treasury.load_or_create(
            initial_capital=config.risk.initial_capital,
            reserve_fraction=config.execution.reserve_fraction,
            state_path=config.execution.treasury_state_path,
        )
        runtime = RuntimeService(config)
        runtime.load_model(model_version)
        executor = build_executor(config, treasury)
        store = PredictionStore(config.logging.db_path) if log_predictions else None
        return cls(config, runtime, executor, treasury, store)

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
        fill = self.executor.rebalance(str(ts), close, pred.decision)

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
                )
            )

        self._prev_close = close
        self._tick_count += 1
        return LiveTickResult(
            prediction=pred,
            fill=fill,
            equity=self.executor.equity,
            trading_capital=self.treasury.trading_capital,
            reserved_wins=self.treasury.reserved_wins,
        )

    def _resolve_outcomes(self, market: pd.DataFrame) -> None:
        """Log realised outcomes once the prediction horizon has elapsed."""
        if self.store is None or not self._pending:
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
        fills = getattr(self.executor, "fills", [])
        fill_count = len(fills) if isinstance(fills, list) else 0
        return LiveSessionResult(
            ticks=self._tick_count,
            fills=fill_count,
            final_equity=self.executor.equity,
            treasury=snapshot,
            model_version=self.runtime.status().model_version or "unknown",
        )

"""Trade executors: paper simulation today, exchange orders when live mode is enabled."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from epoch_ai.config.settings import AppConfig
from epoch_ai.execution.paper_trader import Fill, PaperTrader
from epoch_ai.execution.risk import RiskDecision
from epoch_ai.execution.treasury import Treasury, TreasurySnapshot
from epoch_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class OrderIntent:
    """Describes a target position change the executor should apply."""

    timestamp: str
    symbol: str
    price: float
    target_weight: float
    signal: int


class TradeExecutor(ABC):
    """Bridge from risk decisions to fills (paper or live exchange)."""

    def __init__(self, config: AppConfig, treasury: Treasury) -> None:
        self.config = config
        self.treasury = treasury
        self._session_start_equity = treasury.trading_capital

    @abstractmethod
    def rebalance(self, timestamp: str, price: float, decision: RiskDecision) -> Fill | None:
        """Move toward the target weight at ``price``."""

    @abstractmethod
    def mark_to_market(self, period_return: float) -> float:
        """Apply price movement to the open position."""

    @property
    @abstractmethod
    def equity(self) -> float:
        """Current mark-to-market trading equity."""

    def settle_session(self) -> TreasurySnapshot:
        """Allocate session PnL between reinvestment and reserved wins."""
        pnl = self.equity - self._session_start_equity
        snapshot = self.treasury.allocate_session_pnl(pnl)
        self._session_start_equity = self.treasury.trading_capital
        return snapshot


class PaperExecutor(TradeExecutor):
    """Simulated fills with fees/slippage; uses treasury trading capital."""

    def __init__(self, config: AppConfig, treasury: Treasury) -> None:
        super().__init__(config, treasury)
        risk = config.risk.model_copy(update={"initial_capital": treasury.trading_capital})
        self._trader = PaperTrader(risk)

    def rebalance(self, timestamp: str, price: float, decision: RiskDecision) -> Fill | None:
        return self._trader.rebalance(timestamp, price, decision)

    def mark_to_market(self, period_return: float) -> float:
        return self._trader.mark_to_market(period_return)

    @property
    def equity(self) -> float:
        return self._trader.equity

    @property
    def position_weight(self) -> float:
        return self._trader.position_weight

    @property
    def fills(self) -> list[Fill]:
        return self._trader.fills


class LiveExecutor(PaperExecutor):
    """Live exchange executor.

    When ``execution.live_enabled`` is false or API credentials are missing, orders are
    logged as **dry-run** and routed through the paper simulator so the full pipeline
    can be tested without real capital at risk.
    """

    def __init__(self, config: AppConfig, treasury: Treasury) -> None:
        super().__init__(config, treasury)
        self.execution = config.execution
        self._exchange = None

    def _ensure_exchange(self):
        if self._exchange is not None:
            return self._exchange
        if not self.execution.live_enabled:
            return None
        try:
            import ccxt  # noqa: PLC0415 - optional dependency
        except ImportError as exc:
            raise RuntimeError(
                "ccxt is required for live trading. pip install -r requirements-optional.txt"
            ) from exc

        import os

        api_key = os.environ.get(self.execution.api_key_env, "")
        api_secret = os.environ.get(self.execution.api_secret_env, "")
        if not api_key or not api_secret:
            logger.warning(
                "Live mode enabled but %s/%s not set; dry-run only.",
                self.execution.api_key_env,
                self.execution.api_secret_env,
            )
            return None

        exchange_cls = getattr(ccxt, self.config.data.exchange)
        self._exchange = exchange_cls(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
                "options": {"defaultType": self.config.data.market_type},
            }
        )
        return self._exchange

    def rebalance(self, timestamp: str, price: float, decision: RiskDecision) -> Fill | None:
        exchange = self._ensure_exchange()
        symbol = self.config.primary_symbol
        if exchange is None or self.execution.dry_run:
            logger.info(
                "DRY-RUN order %s @ %.2f -> weight %.3f (signal=%d)",
                symbol,
                price,
                decision.target_weight,
                decision.signal,
            )
            return super().rebalance(timestamp, price, decision)

        # Human: Live orders use market-style rebalance toward target notional weight.
        # Agent: CALLS ccxt create_order when live_enabled; WRITES fill via paper fallback on error.
        try:
            balance = exchange.fetch_balance()
            usdt_free = float(balance.get("USDT", {}).get("free", 0) or 0)
            target_notional = abs(decision.target_weight) * max(usdt_free, self.equity)
            side = "buy" if decision.signal >= 0 else "sell"
            amount = target_notional / price if price > 0 else 0.0
            if amount <= 0:
                return None
            exchange.create_order(symbol, "market", side, amount)
            logger.info(
                "LIVE order placed: %s %s amount=%.6f @ ~%.2f",
                side,
                symbol,
                amount,
                price,
            )
        except Exception as exc:  # noqa: BLE001 - must not crash live loop
            logger.error("Live order failed (%s); falling back to paper fill.", exc)
        return super().rebalance(timestamp, price, decision)


def build_executor(config: AppConfig, treasury: Treasury) -> TradeExecutor:
    """Factory: paper executor unless live mode is explicitly enabled."""
    if config.execution.mode == "live":
        return LiveExecutor(config, treasury)
    return PaperExecutor(config, treasury)

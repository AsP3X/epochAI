"""Paper-trading executor.

Simulates order execution against live/near-real-time prices, applying fees and
slippage, and tracks a simple cash+position portfolio. This is the bridge between the
risk layer and (eventually) a live ``ccxt`` order router that implements the same
interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from epoch_ai.config.settings import RiskConfig
from epoch_ai.execution.risk import RiskDecision
from epoch_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class Fill:
    """A simulated fill."""

    timestamp: str
    price: float
    target_weight: float
    fee: float


@dataclass(slots=True)
class PaperTrader:
    """A minimal paper-trading portfolio with fees and slippage."""

    risk: RiskConfig
    equity: float = field(init=False)
    position_weight: float = field(default=0.0, init=False)
    fills: list[Fill] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.equity = self.risk.initial_capital

    def rebalance(self, timestamp: str, price: float, decision: RiskDecision) -> Fill | None:
        """Move the portfolio toward ``decision.target_weight``.

        Trading costs are charged on the *traded* notional (the change in weight).

        Args:
            timestamp: ISO time of the decision.
            price: Current mark price.
            decision: The desired risk decision.

        Returns:
            The resulting :class:`Fill`, or ``None`` if no trade was needed.
        """
        delta = decision.target_weight - self.position_weight
        if abs(delta) < 1e-9:
            return None

        cost_rate = self.risk.fee_rate + self.risk.slippage
        fee = abs(delta) * self.equity * cost_rate
        self.equity -= fee
        self.position_weight = decision.target_weight

        fill = Fill(timestamp=timestamp, price=price, target_weight=decision.target_weight, fee=fee)
        self.fills.append(fill)
        logger.info(
            "Rebalance @ %s price=%.2f -> weight=%.3f (fee=%.4f)",
            timestamp,
            price,
            decision.target_weight,
            fee,
        )
        return fill

    def mark_to_market(self, period_return: float) -> float:
        """Apply one period's price return to the held position and return equity."""
        self.equity *= 1.0 + self.position_weight * period_return
        return self.equity

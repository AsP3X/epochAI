"""Mutable portfolio snapshot consumed by the risk layer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PortfolioState:
    """Execution-time portfolio metrics for risk halts and cooldowns."""

    equity: float
    peak_equity: float
    session_start_equity: float
    cooldown_remaining: int = 0
    bars_elapsed: int = 0
    position_weight: float = 0.0
    bars_in_position: int = 0

    @classmethod
    def initial(cls, equity: float) -> PortfolioState:
        """Create state at loop start."""
        return cls(
            equity=equity,
            peak_equity=equity,
            session_start_equity=equity,
        )

    def drawdown(self) -> float:
        """Peak-to-trough drawdown as a positive fraction."""
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.equity) / self.peak_equity)

    def session_loss(self) -> float:
        """Loss since loop start as a positive fraction."""
        if self.session_start_equity <= 0:
            return 0.0
        return max(0.0, (self.session_start_equity - self.equity) / self.session_start_equity)

    def after_bar(
        self,
        equity: float,
        *,
        lost_trade: bool = False,
        cooldown_bars: int = 0,
        position_weight: float | None = None,
    ) -> None:
        """Update state after a bar closes."""
        self.equity = equity
        self.peak_equity = max(self.peak_equity, equity)
        self.bars_elapsed += 1
        if position_weight is not None:
            if abs(position_weight) < 1e-9:
                self.bars_in_position = 0
            elif abs(position_weight - self.position_weight) < 1e-9:
                self.bars_in_position += 1
            else:
                self.bars_in_position = 1 if abs(position_weight) > 1e-9 else 0
            self.position_weight = position_weight
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
        if lost_trade and cooldown_bars > 0:
            self.cooldown_remaining = cooldown_bars

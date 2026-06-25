"""Profit allocation: reinvest trading gains vs set wins aside."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from epoch_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class TreasurySnapshot:
    """Balances after allocation."""

    trading_capital: float
    reserved_wins: float
    cold_storage_wins: float
    total_realized_pnl: float
    last_session_pnl: float = 0.0
    last_reserved: float = 0.0
    last_reinvested: float = 0.0
    last_cold_storage: float = 0.0


class Treasury:
    """Track active trading capital separately from reserved and cold storage wins."""

    def __init__(
        self,
        *,
        trading_capital: float,
        reserve_fraction: float = 0.0,
        cold_storage_fraction: float = 0.0,
        max_daily_profit_take: float | None = None,
        state_path: str = "artifacts/treasury.json",
    ) -> None:
        self.trading_capital = trading_capital
        self.reserved_wins = 0.0
        self.cold_storage_wins = 0.0
        self.total_realized_pnl = 0.0
        self.reserve_fraction = min(max(reserve_fraction, 0.0), 1.0)
        self.cold_storage_fraction = min(max(cold_storage_fraction, 0.0), 1.0)
        self.max_daily_profit_take = max_daily_profit_take
        self.state_path = Path(state_path)
        self._daily_take_date: date | None = None
        self._daily_taken = 0.0

    @classmethod
    def load_or_create(
        cls,
        *,
        initial_capital: float,
        reserve_fraction: float,
        cold_storage_fraction: float = 0.0,
        max_daily_profit_take: float | None = None,
        state_path: str,
    ) -> Treasury:
        """Load persisted treasury or create a fresh one."""
        path = Path(state_path)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            treasury = cls(
                trading_capital=float(data["trading_capital"]),
                reserve_fraction=reserve_fraction,
                cold_storage_fraction=cold_storage_fraction,
                max_daily_profit_take=max_daily_profit_take,
                state_path=state_path,
            )
            treasury.reserved_wins = float(data.get("reserved_wins", 0.0))
            treasury.cold_storage_wins = float(data.get("cold_storage_wins", 0.0))
            treasury.total_realized_pnl = float(data.get("total_realized_pnl", 0.0))
            daily_date = data.get("daily_take_date")
            if daily_date:
                treasury._daily_take_date = date.fromisoformat(str(daily_date))
            treasury._daily_taken = float(data.get("daily_taken", 0.0))
            return treasury
        return cls(
            trading_capital=initial_capital,
            reserve_fraction=reserve_fraction,
            cold_storage_fraction=cold_storage_fraction,
            max_daily_profit_take=max_daily_profit_take,
            state_path=state_path,
        )

    def save(self) -> None:
        """Persist treasury state to disk."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "trading_capital": self.trading_capital,
            "reserved_wins": self.reserved_wins,
            "cold_storage_wins": self.cold_storage_wins,
            "total_realized_pnl": self.total_realized_pnl,
            "reserve_fraction": self.reserve_fraction,
            "cold_storage_fraction": self.cold_storage_fraction,
            "daily_take_date": self._daily_take_date.isoformat()
            if self._daily_take_date
            else None,
            "daily_taken": self._daily_taken,
        }
        self.state_path.write_text(json.dumps(payload, indent=2))

    def _reset_daily_take_if_needed(self) -> None:
        today = datetime.now(UTC).date()
        if self._daily_take_date != today:
            self._daily_take_date = today
            self._daily_taken = 0.0

    def _cap_profit_take(self, take_amount: float) -> float:
        """Apply daily profit-withdrawal cap when configured."""
        if take_amount <= 0 or self.max_daily_profit_take is None:
            return take_amount
        self._reset_daily_take_if_needed()
        remaining = max(self.max_daily_profit_take - self._daily_taken, 0.0)
        capped = min(take_amount, remaining)
        self._daily_taken += capped
        return capped

    def allocate_session_pnl(self, session_pnl: float) -> TreasurySnapshot:
        """Split session profit across reinvest, reserve, and cold storage."""
        if session_pnl <= 0:
            self.trading_capital += session_pnl
            self.total_realized_pnl += session_pnl
            self.save()
            return TreasurySnapshot(
                trading_capital=self.trading_capital,
                reserved_wins=self.reserved_wins,
                cold_storage_wins=self.cold_storage_wins,
                total_realized_pnl=self.total_realized_pnl,
                last_session_pnl=session_pnl,
            )

        raw_reserved = session_pnl * self.reserve_fraction
        raw_cold = session_pnl * self.cold_storage_fraction
        raw_take = raw_reserved + raw_cold
        allowed_take = self._cap_profit_take(raw_take)
        scale = allowed_take / raw_take if raw_take > 0 else 0.0
        reserved = raw_reserved * scale
        cold = raw_cold * scale
        reinvested = session_pnl - reserved - cold

        self.reserved_wins += reserved
        self.cold_storage_wins += cold
        self.trading_capital += reinvested
        self.total_realized_pnl += session_pnl
        self.save()
        logger.info(
            "Treasury: session PnL=%.2f | reinvested=%.2f | reserved=%.2f | "
            "cold=%.2f | total reserved=%.2f | cold storage=%.2f",
            session_pnl,
            reinvested,
            reserved,
            cold,
            self.reserved_wins,
            self.cold_storage_wins,
        )
        return TreasurySnapshot(
            trading_capital=self.trading_capital,
            reserved_wins=self.reserved_wins,
            cold_storage_wins=self.cold_storage_wins,
            total_realized_pnl=self.total_realized_pnl,
            last_session_pnl=session_pnl,
            last_reserved=reserved,
            last_reinvested=reinvested,
            last_cold_storage=cold,
        )

    @property
    def total_wealth(self) -> float:
        """Combined trading + reserved + cold storage capital."""
        return self.trading_capital + self.reserved_wins + self.cold_storage_wins

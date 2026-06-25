"""Profit allocation: reinvest trading gains vs set wins aside."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from epoch_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class TreasurySnapshot:
    """Balances after allocation."""

    trading_capital: float
    reserved_wins: float
    total_realized_pnl: float
    last_session_pnl: float = 0.0
    last_reserved: float = 0.0
    last_reinvested: float = 0.0


class Treasury:
    """Track active trading capital separately from reserved (set-aside) wins."""

    def __init__(
        self,
        *,
        trading_capital: float,
        reserve_fraction: float = 0.0,
        state_path: str = "artifacts/treasury.json",
    ) -> None:
        self.trading_capital = trading_capital
        self.reserved_wins = 0.0
        self.total_realized_pnl = 0.0
        self.reserve_fraction = min(max(reserve_fraction, 0.0), 1.0)
        self.state_path = Path(state_path)

    @classmethod
    def load_or_create(
        cls,
        *,
        initial_capital: float,
        reserve_fraction: float,
        state_path: str,
    ) -> Treasury:
        """Load persisted treasury or create a fresh one."""
        path = Path(state_path)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            treasury = cls(
                trading_capital=float(data["trading_capital"]),
                reserve_fraction=reserve_fraction,
                state_path=state_path,
            )
            treasury.reserved_wins = float(data.get("reserved_wins", 0.0))
            treasury.total_realized_pnl = float(data.get("total_realized_pnl", 0.0))
            return treasury
        return cls(
            trading_capital=initial_capital,
            reserve_fraction=reserve_fraction,
            state_path=state_path,
        )

    def save(self) -> None:
        """Persist treasury state to disk."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "trading_capital": self.trading_capital,
            "reserved_wins": self.reserved_wins,
            "total_realized_pnl": self.total_realized_pnl,
            "reserve_fraction": self.reserve_fraction,
        }
        self.state_path.write_text(json.dumps(payload, indent=2))

    def allocate_session_pnl(self, session_pnl: float) -> TreasurySnapshot:
        """Split session profit: reinvest into trading capital, reserve the rest."""
        if session_pnl <= 0:
            self.trading_capital += session_pnl
            self.total_realized_pnl += session_pnl
            self.save()
            return TreasurySnapshot(
                trading_capital=self.trading_capital,
                reserved_wins=self.reserved_wins,
                total_realized_pnl=self.total_realized_pnl,
                last_session_pnl=session_pnl,
            )

        reserved = session_pnl * self.reserve_fraction
        reinvested = session_pnl - reserved
        self.reserved_wins += reserved
        self.trading_capital += reinvested
        self.total_realized_pnl += session_pnl
        self.save()
        logger.info(
            "Treasury: session PnL=%.2f | reinvested=%.2f | reserved=%.2f | total reserved=%.2f",
            session_pnl,
            reinvested,
            reserved,
            self.reserved_wins,
        )
        return TreasurySnapshot(
            trading_capital=self.trading_capital,
            reserved_wins=self.reserved_wins,
            total_realized_pnl=self.total_realized_pnl,
            last_session_pnl=session_pnl,
            last_reserved=reserved,
            last_reinvested=reinvested,
        )

    @property
    def total_wealth(self) -> float:
        """Combined trading + reserved capital."""
        return self.trading_capital + self.reserved_wins

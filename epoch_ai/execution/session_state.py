"""Persist/resume paper-trading session state across restarts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from epoch_ai.execution.portfolio_state import PortfolioState


@dataclass(slots=True)
class SessionState:
    """Serializable open-position and equity snapshot."""

    equity: float
    peak_equity: float
    session_start_equity: float
    position_weight: float
    bars_in_position: int
    cooldown_remaining: int
    bars_elapsed: int

    @classmethod
    def from_portfolio(cls, portfolio: PortfolioState) -> SessionState:
        return cls(
            equity=portfolio.equity,
            peak_equity=portfolio.peak_equity,
            session_start_equity=portfolio.session_start_equity,
            position_weight=portfolio.position_weight,
            bars_in_position=portfolio.bars_in_position,
            cooldown_remaining=portfolio.cooldown_remaining,
            bars_elapsed=portfolio.bars_elapsed,
        )

    def to_portfolio(self) -> PortfolioState:
        return PortfolioState(
            equity=self.equity,
            peak_equity=self.peak_equity,
            session_start_equity=self.session_start_equity,
            cooldown_remaining=self.cooldown_remaining,
            bars_elapsed=self.bars_elapsed,
            position_weight=self.position_weight,
            bars_in_position=self.bars_in_position,
        )

    @classmethod
    def load(cls, path: str | Path) -> SessionState | None:
        path = Path(path)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

"""Global trading halt flag — checked before every live trade."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(slots=True)
class KillSwitchState:
    halted: bool
    reason: str = ""
    updated_at: str = ""


class KillSwitch:
    """File-backed kill switch shared by API, CLI, and live engine."""

    def __init__(self, path: str = "artifacts/kill_switch.json") -> None:
        self.path = Path(path)

    def read(self) -> KillSwitchState:
        if not self.path.exists():
            return KillSwitchState(halted=False)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return KillSwitchState(
            halted=bool(data.get("halted", False)),
            reason=str(data.get("reason", "")),
            updated_at=str(data.get("updated_at", "")),
        )

    def is_halted(self) -> bool:
        return self.read().halted

    def halt(self, reason: str = "manual halt") -> KillSwitchState:
        state = KillSwitchState(
            halted=True,
            reason=reason,
            updated_at=datetime.now(UTC).isoformat(),
        )
        self._write(state)
        return state

    def resume(self) -> KillSwitchState:
        state = KillSwitchState(
            halted=False,
            reason="",
            updated_at=datetime.now(UTC).isoformat(),
        )
        self._write(state)
        return state

    def _write(self, state: KillSwitchState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "halted": state.halted,
                    "reason": state.reason,
                    "updated_at": state.updated_at,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

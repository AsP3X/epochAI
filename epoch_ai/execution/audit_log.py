"""Append-only JSONL audit trail for trade decisions."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class AuditLog:
    """Immutable-style event log for predictions, fills, and halts."""

    def __init__(self, path: str = "artifacts/audit/trades.jsonl") -> None:
        self.path = Path(path)

    def append(self, event_type: str, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "type": event_type,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

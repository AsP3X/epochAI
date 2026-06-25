"""Lightweight JSON metrics recorder for dashboards."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class MetricsRecorder:
    """Append timestamped metric snapshots to a JSON lines file."""

    def __init__(self, path: str = "artifacts/metrics/runtime.jsonl") -> None:
        self.path = Path(path)

    def record(self, name: str, values: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ts": datetime.now(UTC).isoformat(),
            "name": name,
            **values,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")

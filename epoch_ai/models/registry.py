"""Lightweight, file-based model registry with versioning.

Each saved model gets a monotonically increasing version directory containing the
booster file and a JSON metadata sidecar (timestamp, train range, metrics, feature
count). This gives reproducible, inspectable lineage without external infrastructure.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from epoch_ai.models.lightgbm_model import LightGBMModel
from epoch_ai.utils.logging import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """Persist and retrieve versioned models under a base directory."""

    def __init__(self, base_dir: str) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _next_version(self) -> int:
        versions = [
            int(p.name.split("_")[-1])
            for p in self.base_dir.glob("v_*")
            if p.name.split("_")[-1].isdigit()
        ]
        return (max(versions) + 1) if versions else 1

    def save(self, model: LightGBMModel, metadata: dict[str, Any] | None = None) -> str:
        """Save ``model`` as a new version and return its version label."""
        version = self._next_version()
        label = f"v_{version}"
        version_dir = self.base_dir / label
        version_dir.mkdir(parents=True, exist_ok=True)

        model.save(str(version_dir / "model.txt"))
        meta = {
            "version": version,
            "label": label,
            "task": model.task,
            "created_at": datetime.now(UTC).isoformat(),
            "best_iteration": model.best_iteration_,
            "n_features": len(model.feature_names_ or []),
            **(metadata or {}),
        }
        (version_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
        logger.info("Registered model %s (%d features).", label, meta["n_features"])
        return label

    def latest_label(self) -> str | None:
        """Return the highest existing version label, or ``None`` if empty."""
        versions = [
            int(p.name.split("_")[-1])
            for p in self.base_dir.glob("v_*")
            if p.name.split("_")[-1].isdigit()
        ]
        return f"v_{max(versions)}" if versions else None

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

from epoch_ai.config.settings import ModelConfig
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

    def list_versions(self) -> list[dict[str, Any]]:
        """Return metadata dicts for every registered version, oldest first."""
        entries: list[dict[str, Any]] = []
        for path in sorted(self.base_dir.glob("v_*")):
            meta_path = path / "metadata.json"
            if meta_path.exists():
                entries.append(json.loads(meta_path.read_text(encoding="utf-8")))
        return entries

    def load(
        self,
        label: str | None,
        config: ModelConfig,
        *,
        task: str = "classification",
    ) -> tuple[LightGBMModel, dict[str, Any]]:
        """Load a versioned model and its metadata sidecar."""
        resolved = label or self.latest_label()
        if not resolved:
            raise FileNotFoundError(
                f"No models in registry at {self.base_dir}. Run training first."
            )
        version_dir = self.base_dir / resolved
        model_path = version_dir / "model.txt"
        meta_path = version_dir / "metadata.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file missing for {resolved}: {model_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        model_task = str(meta.get("task", task))
        model = LightGBMModel.load(str(model_path), config, task=model_task)
        return model, meta

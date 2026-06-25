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
from epoch_ai.models.base import BaseModel
from epoch_ai.models.factory import model_class
from epoch_ai.utils.logging import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """Persist and retrieve versioned models under a base directory.

    Besides the immutable ``v_*`` version directories, the registry tracks a single
    **promoted** ("champion") pointer in ``current.json``. Runtime model resolution
    prefers the promoted label so an automated retrain can register a new version
    without it going live until it passes the promotion gate (see
    :mod:`epoch_ai.learning.promotion`). When no pointer exists the registry falls
    back to the latest version, preserving the original behaviour.
    """

    def __init__(self, base_dir: str) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _pointer_path(self) -> Path:
        return self.base_dir / "current.json"

    def _model_path(self, label: str) -> Path | None:
        """Return the booster file for ``label`` (backend-aware), or ``None``.

        The booster filename depends on the backend (``model.txt`` for LightGBM,
        ``model.json`` for XGBoost). It is recorded in each version's metadata; older
        versions without the key default to ``model.txt`` for backward compatibility.
        """
        meta_path = self.base_dir / label / "metadata.json"
        model_file = "model.txt"
        if meta_path.exists():
            try:
                model_file = json.loads(meta_path.read_text(encoding="utf-8")).get(
                    "model_file", "model.txt"
                )
            except (json.JSONDecodeError, OSError):
                model_file = "model.txt"
        path = self.base_dir / label / model_file
        return path if path.exists() else None

    def promoted_label(self) -> str | None:
        """Return the currently promoted ("champion") label, if one is set and valid."""
        pointer = self._pointer_path
        if not pointer.exists():
            return None
        try:
            label = json.loads(pointer.read_text(encoding="utf-8")).get("label")
        except (json.JSONDecodeError, OSError):
            return None
        if label and self._model_path(label) is not None:
            return label
        return None

    def set_promoted(self, label: str, *, info: dict[str, Any] | None = None) -> None:
        """Point the champion at ``label`` (must be an existing version)."""
        if self._model_path(label) is None:
            raise FileNotFoundError(f"Cannot promote unknown version: {label}")
        payload = {
            "label": label,
            "promoted_at": datetime.now(UTC).isoformat(),
            **(info or {}),
        }
        self._pointer_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        logger.info("Promoted %s to champion.", label)

    def resolve_label(self, label: str | None) -> str | None:
        """Resolve an explicit label, else the promoted champion, else the latest."""
        return label or self.promoted_label() or self.latest_label()

    def _next_version(self) -> int:
        versions = [
            int(p.name.split("_")[-1])
            for p in self.base_dir.glob("v_*")
            if p.name.split("_")[-1].isdigit()
        ]
        return (max(versions) + 1) if versions else 1

    def save(self, model: BaseModel, metadata: dict[str, Any] | None = None) -> str:
        """Save ``model`` as a new version and return its version label."""
        version = self._next_version()
        label = f"v_{version}"
        version_dir = self.base_dir / label
        version_dir.mkdir(parents=True, exist_ok=True)

        model_file = getattr(model, "MODEL_FILENAME", "model.txt")
        model.save(str(version_dir / model_file))
        meta = {
            "version": version,
            "label": label,
            "task": model.task,
            "backend": getattr(model, "BACKEND", "lightgbm"),
            "model_file": model_file,
            "created_at": datetime.now(UTC).isoformat(),
            "best_iteration": model.best_iteration_,
            "n_features": len(model.feature_names_ or []),
            "open_weights": True,
            "feature_names": list(model.feature_names_ or []),
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
    ) -> tuple[BaseModel, dict[str, Any]]:
        """Load a versioned model and its metadata sidecar.

        The backend is read from metadata (``backend``/``model_file``) so a registry can
        mix LightGBM and XGBoost versions. When ``label`` is ``None`` the promoted
        champion is used if one is set, otherwise the latest version.
        """
        resolved = self.resolve_label(label)
        if not resolved:
            raise FileNotFoundError(
                f"No models in registry at {self.base_dir}. Run training first."
            )
        version_dir = self.base_dir / resolved
        meta_path = version_dir / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        backend = str(meta.get("backend", "lightgbm"))
        model_file = str(meta.get("model_file", "model.txt"))
        model_path = version_dir / model_file
        if not model_path.exists():
            raise FileNotFoundError(f"Model file missing for {resolved}: {model_path}")
        model_task = str(meta.get("task", task))
        model = model_class(backend).load(str(model_path), config, task=model_task)
        return model, meta

    def export_open_bundle(self, dest: str | Path, label: str | None = None) -> Path:
        """Copy a versioned model into a portable open-weights bundle directory.

        The bundle contains plain ``model.txt``, ``metadata.json``, and a small
        ``README.txt`` describing reproducibility — no encryption or license gates.

        Args:
            dest: Output directory (created if missing).
            label: Registry version to export; latest when ``None``.

        Returns:
            Path to the bundle root.
        """
        import shutil

        resolved = label or self.latest_label()
        if not resolved:
            raise FileNotFoundError("No models to export.")

        src = self.base_dir / resolved
        out = Path(dest) / resolved
        out.mkdir(parents=True, exist_ok=True)

        # Resolve the backend-specific booster filename from metadata.
        meta_path = src / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        backend = str(meta.get("backend", "lightgbm"))
        model_file = str(meta.get("model_file", "model.txt"))

        # Include the calibration sidecar so the exported bundle predicts identically.
        for name in (model_file, f"{model_file}.calibration.json", "metadata.json"):
            src_file = src / name
            if src_file.exists():
                shutil.copy2(src_file, out / name)

        fmt = "XGBoost JSON booster" if backend == "xgboost" else "LightGBM text booster"
        loader = (
            "epoch_ai.models.registry.ModelRegistry(...).load(label, cfg.model)"
        )
        readme = out / "README.txt"
        readme.write_text(
            "epochAI open-weights bundle\n"
            f"version: {resolved}\n"
            f"backend: {backend}\n"
            f"format: {fmt} + JSON metadata\n"
            "license: not specified in this bundle — see repository owner\n"
            f"load: {loader}\n",
            encoding="utf-8",
        )
        logger.info("Exported open-weights bundle to %s", out)
        return out

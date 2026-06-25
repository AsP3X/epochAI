"""Thin MLflow wrapper with a no-op fallback.

When MLflow is installed and tracking is enabled in config, metrics/params are logged
to the configured tracking URI. Otherwise every method is a safe no-op, so callers
never need to branch on availability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from epoch_ai.config.settings import TrackingConfig
from epoch_ai.utils.logging import get_logger

logger = get_logger(__name__)


class MLflowTracker:
    """Log params/metrics to MLflow when available; otherwise do nothing."""

    def __init__(self, config: TrackingConfig) -> None:
        self.config = config
        self._mlflow = None
        if not config.enabled:
            return
        try:
            import mlflow  # noqa: PLC0415 - optional dependency
        except ImportError:
            logger.warning("mlflow not installed; tracking disabled.")
            return
        mlflow.set_tracking_uri(config.tracking_uri)
        mlflow.set_experiment(config.experiment_name)
        self._mlflow = mlflow

    @property
    def active(self) -> bool:
        """Whether MLflow logging is active."""
        return self._mlflow is not None

    def __enter__(self) -> MLflowTracker:
        if self._mlflow is not None:
            self._mlflow.start_run()
        return self

    def __exit__(self, *exc) -> None:
        if self._mlflow is not None:
            self._mlflow.end_run()

    def log_params(self, params: dict[str, Any]) -> None:
        """Log a dict of parameters."""
        if self._mlflow is not None:
            flat = {k: str(v) for k, v in params.items()}
            self._mlflow.log_params(flat)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Log a dict of numeric metrics."""
        if self._mlflow is not None:
            self._mlflow.log_metrics({k: float(v) for k, v in metrics.items()})

    def log_learning_metrics(
        self,
        step_history: pd.DataFrame,
        learning_improvement: dict[str, float],
        learning_curve: dict[str, float],
    ) -> None:
        """Log walk-forward learning diagnostics."""
        if self._mlflow is None:
            return
        self.log_metrics({f"learning_{k}": v for k, v in learning_improvement.items()})
        self.log_metrics({f"curve_{k}": v for k, v in learning_curve.items()})
        if not step_history.empty and "oos_accuracy" in step_history.columns:
            self.log_metrics({"final_step_oos_accuracy": float(step_history["oos_accuracy"].iloc[-1])})

    def log_artifact(self, path: str | Path) -> None:
        """Log a file artifact when MLflow is active."""
        if self._mlflow is not None:
            self._mlflow.log_artifact(str(path))

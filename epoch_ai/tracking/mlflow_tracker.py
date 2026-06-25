"""Thin MLflow wrapper with a no-op fallback.

When MLflow is installed and tracking is enabled in config, metrics/params are logged
to the configured tracking URI. Otherwise every method is a safe no-op, so callers
never need to branch on availability.
"""

from __future__ import annotations

from typing import Any

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
            self._mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Log a dict of numeric metrics."""
        if self._mlflow is not None:
            self._mlflow.log_metrics({k: float(v) for k, v in metrics.items()})

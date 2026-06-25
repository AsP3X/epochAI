"""Tests for MLflow tracker no-op and mocked active path."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from epoch_ai.config.settings import TrackingConfig
from epoch_ai.tracking.mlflow_tracker import MLflowTracker


def test_tracker_disabled_is_noop():
    tracker = MLflowTracker(TrackingConfig(enabled=False))
    assert not tracker.active
    with tracker:
        tracker.log_params({"a": 1})
        tracker.log_metrics({"b": 2.0})
        tracker.log_learning_metrics(pd.DataFrame(), {}, {})
        tracker.log_artifact(Path("/tmp/x"))


def test_tracker_active_logs():
    mock_mlflow = MagicMock()
    cfg = TrackingConfig(enabled=True)
    with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
        tracker = MLflowTracker(cfg)
    assert tracker.active
    with tracker:
        tracker.log_params({"symbol": "BTC/USDT"})
        tracker.log_metrics({"sharpe": 1.2})
        tracker.log_learning_metrics(
            pd.DataFrame({"oos_accuracy": [0.5]}),
            {"delta": 0.01},
            {"n_steps": 1.0},
        )
        tracker.log_artifact("/tmp/metrics.json")
    mock_mlflow.start_run.assert_called_once()
    mock_mlflow.end_run.assert_called_once()

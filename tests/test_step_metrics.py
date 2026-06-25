"""Tests for per-step out-of-sample metrics."""

from __future__ import annotations

import numpy as np

from epoch_ai.learning.step_metrics import (
    classification_step_metrics,
    regression_step_metrics,
)


def test_classification_metrics_perfect_separation():
    preds = np.array([0.9, 0.8, 0.1, 0.2])
    labels = np.array([1, 1, 0, 0])
    m = classification_step_metrics(preds, labels, long_threshold=0.55, short_threshold=0.45)
    assert m["oos_accuracy"] == 1.0
    assert m["oos_auc"] == 1.0
    assert m["oos_brier"] < 0.1
    # All four bars trigger a directional signal and are correct.
    assert m["oos_coverage"] == 1.0
    assert m["oos_directional_accuracy"] == 1.0


def test_classification_auc_nan_single_class():
    preds = np.array([0.6, 0.7, 0.8])
    labels = np.array([1, 1, 1])
    m = classification_step_metrics(preds, labels, long_threshold=0.55, short_threshold=0.45)
    assert np.isnan(m["oos_auc"])


def test_coverage_excludes_neutral_band():
    # Predictions inside the neutral band do not trigger a directional signal.
    preds = np.array([0.5, 0.5, 0.9, 0.1])
    labels = np.array([1, 0, 1, 0])
    m = classification_step_metrics(preds, labels, long_threshold=0.55, short_threshold=0.45)
    assert m["oos_coverage"] == 0.5
    assert m["oos_directional_accuracy"] == 1.0


def test_regression_metrics_directional():
    preds = np.array([0.01, -0.02, 0.03, -0.01])
    rets = np.array([0.02, -0.01, -0.01, -0.02])
    m = regression_step_metrics(preds, rets)
    assert m["oos_accuracy"] == 0.75  # 3 of 4 signs agree
    assert m["oos_rmse"] > 0

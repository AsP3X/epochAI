"""Multi-horizon step metrics and calibrator helpers."""

from __future__ import annotations

import numpy as np

from epoch_ai.learning.step_metrics import (
    confidence_weighted_brier,
    multi_horizon_classification_step_metrics,
)
from epoch_ai.models.calibration import (
    MultiHeadCalibrator,
    coverage_reliability,
    load_calibrator_sidecar,
    quantile_interval_coverage,
)


def test_quantile_coverage_and_reliability():
    realized = np.array([-0.01, 0.0, 0.02, 0.05])
    q_low = np.array([-0.02, -0.01, -0.01, 0.0])
    q_high = np.array([0.01, 0.01, 0.03, 0.04])
    assert quantile_interval_coverage(realized, q_low, q_high) == 0.75
    assert coverage_reliability(0.8) == 1.0
    assert coverage_reliability(0.4) == 0.5


def test_confidence_weighted_brier():
    briers = {1: 0.2, 5: 0.3}
    weights = {1: 1.0, 5: 0.0}
    assert confidence_weighted_brier(briers, weights) == 0.2
    weights = {1: 1.0, 5: 1.0}
    assert confidence_weighted_brier(briers, weights) == 0.25


def test_multi_horizon_step_metrics_keys():
    n = 40
    rng = np.random.default_rng(0)
    structured = {
        1: {
            "p_up": rng.uniform(0.3, 0.7, n),
            "q10": rng.normal(-0.01, 0.002, n),
            "q50": rng.normal(0.0, 0.002, n),
            "q90": rng.normal(0.01, 0.002, n),
        },
        5: {
            "p_up": rng.uniform(0.3, 0.7, n),
            "q10": rng.normal(-0.02, 0.003, n),
            "q50": rng.normal(0.0, 0.003, n),
            "q90": rng.normal(0.02, 0.003, n),
        },
    }
    labels = {1: (rng.uniform(0, 1, n) > 0.5).astype(float), 5: (rng.uniform(0, 1, n) > 0.5).astype(float)}
    returns = {1: rng.normal(0, 0.01, n), 5: rng.normal(0, 0.02, n)}
    metrics = multi_horizon_classification_step_metrics(
        structured,
        labels,
        returns,
        long_threshold=0.58,
        short_threshold=0.42,
        primary_horizon=5,
    )
    assert "oos_brier_weighted" in metrics
    assert "oos_brier_h1" in metrics
    assert "oos_coverage_h5" in metrics
    assert "oos_brier" in metrics


def test_multi_head_calibrator_roundtrip():
    rng = np.random.default_rng(1)
    n = 120
    raw = {1: rng.uniform(0.2, 0.8, n), 5: rng.uniform(0.2, 0.8, n)}
    labels = {
        1: (rng.uniform(0, 1, n) > 0.55).astype(float),
        5: (rng.uniform(0, 1, n) > 0.45).astype(float),
    }
    mh = MultiHeadCalibrator.fit(raw, labels, (1, 5), "isotonic")
    payload = mh.to_dict()
    restored = load_calibrator_sidecar(payload)
    assert isinstance(restored, MultiHeadCalibrator)
    for h in (1, 5):
        np.testing.assert_allclose(
            mh.transform(h, raw[h]),
            restored.transform(h, raw[h]),
        )

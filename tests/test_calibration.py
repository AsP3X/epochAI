"""Tests for calibration tracker."""

from __future__ import annotations

from epoch_ai.calibration.tracker import CalibrationTracker


def test_calibration_gate_passes_with_insufficient_samples():
    tracker = CalibrationTracker(min_accuracy=0.6, min_samples=30)
    for pred, label in [(0.7, 1), (0.3, 0), (0.8, 1)]:
        tracker.record(pred, label)
    gate = tracker.check_gate()
    assert gate.passed
    assert gate.n_samples == 3


def test_calibration_gate_blocks_when_accuracy_low():
    tracker = CalibrationTracker(min_accuracy=0.6, min_samples=5)
    samples = [(0.9, 0), (0.9, 0), (0.9, 0), (0.9, 0), (0.9, 0), (0.1, 1)]
    for pred, label in samples:
        tracker.record(pred, label)
    gate = tracker.check_gate()
    assert not gate.passed
    assert gate.n_samples == 6

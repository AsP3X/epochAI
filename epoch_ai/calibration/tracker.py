"""Track predicted vs realised outcomes for calibration gating."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class CalibrationGate:
    """Result of a calibration check before live trading."""

    passed: bool
    n_samples: int
    mean_accuracy: float
    brier_score: float
    min_accuracy_required: float | None


@dataclass
class CalibrationTracker:
    """Rolling window of classification predictions vs labels."""

    min_accuracy: float | None = None
    min_samples: int = 30
    _preds: list[float] = field(default_factory=list)
    _labels: list[int] = field(default_factory=list)

    def record(self, prediction: float, label: int) -> None:
        self._preds.append(float(prediction))
        self._labels.append(int(label))

    @property
    def n_samples(self) -> int:
        return len(self._labels)

    def mean_accuracy(self) -> float:
        if not self._labels:
            return 0.0
        hits = sum(int(p >= 0.5) == bool(y) for p, y in zip(self._preds, self._labels, strict=True))
        return hits / len(self._labels)

    def brier_score(self) -> float:
        if not self._labels:
            return 1.0
        p = np.clip(np.array(self._preds), 1e-6, 1 - 1e-6)
        y = np.array(self._labels, dtype=float)
        return float(np.mean((p - y) ** 2))

    def check_gate(self) -> CalibrationGate:
        acc = self.mean_accuracy()
        brier = self.brier_score()
        if self.min_accuracy is None:
            passed = True
        elif self.n_samples < self.min_samples:
            passed = True
        else:
            passed = acc >= self.min_accuracy
        return CalibrationGate(
            passed=passed,
            n_samples=self.n_samples,
            mean_accuracy=acc,
            brier_score=brier,
            min_accuracy_required=self.min_accuracy,
        )

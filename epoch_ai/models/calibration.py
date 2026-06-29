"""Post-hoc probability calibration for classification models.

Raw LightGBM scores are not guaranteed to be well-calibrated probabilities, yet the
execution layer thresholds on P(up) (e.g. ``long_threshold=0.55``). Calibrating the
model output on a held-out validation tail makes those thresholds meaningful.

Two methods are supported:

* ``"isotonic"`` - non-parametric, monotone step function (flexible, needs more data).
* ``"sigmoid"``  - Platt scaling: a logistic curve fit on the raw probabilities.

Fitting uses scikit-learn (a core dependency), but :meth:`transform` is implemented in
pure numpy from the serialized parameters, so a *loaded* model never needs to
reconstruct a scikit-learn estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

_EPS = 1e-6


@dataclass(slots=True)
class ProbabilityCalibrator:
    """A fitted 1-D probability calibrator (isotonic or Platt/sigmoid).

    Attributes:
        method: ``"isotonic"`` or ``"sigmoid"``.
        xs / ys: Isotonic interpolation knots (only for ``"isotonic"``).
        a / b: Logistic slope/intercept on the logit of the raw probability
            (only for ``"sigmoid"``).
    """

    method: str
    xs: list[float] | None = None
    ys: list[float] | None = None
    a: float | None = None
    b: float | None = None

    # --------------------------------------------------------------------- fit
    @classmethod
    def fit(
        cls, raw: np.ndarray, labels: np.ndarray, method: str
    ) -> ProbabilityCalibrator | None:
        """Fit a calibrator mapping raw P(up) -> calibrated P(up).

        Args:
            raw: Raw model probabilities on a held-out set.
            labels: Binary {0,1} outcomes aligned to ``raw``.
            method: ``"isotonic"`` or ``"sigmoid"``.

        Returns:
            A fitted calibrator, or ``None`` if calibration is not possible (too few
            samples or only one class present in ``labels``).
        """
        raw = np.asarray(raw, dtype=float).ravel()
        labels = np.asarray(labels, dtype=float).ravel()
        # Human: Calibration is meaningless without both classes or with too little data.
        # Agent: GUARD returns None -> caller keeps raw probabilities.
        if raw.size < 20 or len(np.unique(labels)) < 2:
            return None

        if method == "isotonic":
            from sklearn.isotonic import IsotonicRegression  # noqa: PLC0415 - lazy

            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(raw, labels)
            # Persist the interpolation knots so transform() needs only numpy.
            xs = np.asarray(iso.X_thresholds_, dtype=float)
            ys = np.asarray(iso.y_thresholds_, dtype=float)
            if xs.size < 2:
                return None
            return cls(method="isotonic", xs=xs.tolist(), ys=ys.tolist())

        if method == "sigmoid":
            from sklearn.linear_model import LogisticRegression  # noqa: PLC0415 - lazy

            # Platt scaling fits a logistic curve on the logit of the raw probability.
            logit = _logit(raw).reshape(-1, 1)
            lr = LogisticRegression(C=1e6, solver="lbfgs")
            lr.fit(logit, labels.astype(int))
            return cls(method="sigmoid", a=float(lr.coef_[0][0]), b=float(lr.intercept_[0]))

        raise ValueError(f"Unknown calibration method: {method!r}")

    # ---------------------------------------------------------------- transform
    def transform(self, raw: np.ndarray) -> np.ndarray:
        """Map raw probabilities to calibrated probabilities in ``[0, 1]``."""
        raw = np.asarray(raw, dtype=float).ravel()
        if self.method == "isotonic":
            xs = np.asarray(self.xs, dtype=float)
            ys = np.asarray(self.ys, dtype=float)
            # np.interp clamps to the endpoint values outside [xs[0], xs[-1]].
            out = np.interp(raw, xs, ys)
        elif self.method == "sigmoid":
            out = _sigmoid(self.a * _logit(raw) + self.b)
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unknown calibration method: {self.method!r}")
        return np.clip(out, 0.0, 1.0)

    # -------------------------------------------------------------- (de)serialize
    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict for the registry sidecar."""
        return {"method": self.method, "xs": self.xs, "ys": self.ys, "a": self.a, "b": self.b}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProbabilityCalibrator:
        """Reconstruct from :meth:`to_dict` output."""
        return cls(
            method=str(data["method"]),
            xs=data.get("xs"),
            ys=data.get("ys"),
            a=data.get("a"),
            b=data.get("b"),
        )


@dataclass(slots=True)
class MultiHeadCalibrator:
    """Per-horizon probability calibrators for multi-head classification models."""

    horizons: tuple[int, ...]
    calibrators: dict[int, ProbabilityCalibrator | None]

    @classmethod
    def fit(
        cls,
        raw_by_horizon: dict[int, np.ndarray],
        labels_by_horizon: dict[int, np.ndarray],
        horizons: tuple[int, ...],
        method: str,
    ) -> MultiHeadCalibrator:
        """Fit one calibrator per horizon (``None`` when fit is not possible)."""
        fitted: dict[int, ProbabilityCalibrator | None] = {}
        for h in horizons:
            raw = np.asarray(raw_by_horizon[h], dtype=float).ravel()
            labels = np.asarray(labels_by_horizon[h], dtype=float).ravel()
            fitted[h] = ProbabilityCalibrator.fit(raw, labels, method)
        return cls(horizons=horizons, calibrators=fitted)

    def transform(self, horizon: int, raw: np.ndarray) -> np.ndarray:
        """Calibrate raw P(up) for one horizon; passthrough when no calibrator exists."""
        raw = np.asarray(raw, dtype=float).ravel()
        cal = self.calibrators.get(horizon)
        if cal is None:
            return np.clip(raw, 0.0, 1.0)
        return cal.transform(raw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "multi_head",
            "horizons": list(self.horizons),
            "per_horizon": {
                str(h): (self.calibrators[h].to_dict() if self.calibrators[h] else None)
                for h in self.horizons
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MultiHeadCalibrator:
        horizons = tuple(int(h) for h in data["horizons"])
        per = data.get("per_horizon") or {}
        calibrators: dict[int, ProbabilityCalibrator | None] = {}
        for h in horizons:
            payload = per.get(str(h))
            calibrators[h] = (
                ProbabilityCalibrator.from_dict(payload) if payload is not None else None
            )
        return cls(horizons=horizons, calibrators=calibrators)


def load_calibrator_sidecar(data: dict[str, Any]) -> ProbabilityCalibrator | MultiHeadCalibrator:
    """Load either a single-head or multi-head calibrator sidecar."""
    if data.get("type") == "multi_head" or "per_horizon" in data:
        return MultiHeadCalibrator.from_dict(data)
    return ProbabilityCalibrator.from_dict(data)


def quantile_interval_coverage(
    realized: np.ndarray,
    q_low: np.ndarray,
    q_high: np.ndarray,
) -> float:
    """Fraction of samples where ``realized`` falls in ``[q_low, q_high]``."""
    realized = np.asarray(realized, dtype=float)
    q_low = np.asarray(q_low, dtype=float)
    q_high = np.asarray(q_high, dtype=float)
    if realized.size == 0:
        return float("nan")
    return float(np.mean((realized >= q_low) & (realized <= q_high)))


def coverage_reliability(coverage: float, *, target: float = 0.8) -> float:
    """Reliability weight from quantile coverage (1.0 when coverage equals ``target``)."""
    if coverage is None or np.isnan(coverage):
        return 0.0
    return max(0.0, 1.0 - abs(coverage - target) / target)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))

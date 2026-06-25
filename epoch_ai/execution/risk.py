"""Risk-management layer.

This is deliberately decoupled from the model: it consumes a model output plus market
state and decides *what to do about it* - direction, confidence and position size.
Swapping the model never requires touching risk logic, and vice versa.
"""

from __future__ import annotations

from dataclasses import dataclass

from epoch_ai.config.settings import PredictionConfig, RiskConfig


@dataclass(slots=True)
class RiskDecision:
    """A risk-adjusted trading decision.

    Attributes:
        signal: Direction (1 long, -1 short, 0 flat).
        confidence: Model confidence in ``[0, 1]``.
        target_weight: Signed fraction of capital to allocate (already leverage- and
            risk-scaled); ``+0.5`` means a 50%-of-capital long.
    """

    signal: int
    confidence: float
    target_weight: float


class RiskManager:
    """Translate model predictions into sized, risk-constrained positions."""

    def __init__(self, risk: RiskConfig, prediction: PredictionConfig) -> None:
        self.risk = risk
        self.prediction = prediction

    def confidence(self, prediction: float) -> float:
        """Map a raw model output to a confidence score in ``[0, 1]``.

        For classification, confidence is how far P(up) is from 0.5 (scaled to
        ``[0, 1]``). For regression it is a saturating function of return magnitude.
        """
        if self.prediction.task == "classification":
            return min(1.0, abs(prediction - 0.5) * 2.0)
        return min(1.0, abs(prediction) / 0.02)

    def decide(self, prediction: float) -> RiskDecision:
        """Produce a :class:`RiskDecision` from a single model output.

        Args:
            prediction: P(up) (classification) or expected return (regression).

        Returns:
            A sized, direction-aware :class:`RiskDecision`.
        """
        conf = self.confidence(prediction)

        if self.prediction.task == "classification":
            if prediction >= self.risk.long_threshold:
                signal = 1
            elif prediction <= self.risk.short_threshold and self.risk.allow_short:
                signal = -1
            else:
                signal = 0
        else:
            if prediction > 0:
                signal = 1
            elif prediction < 0 and self.risk.allow_short:
                signal = -1
            else:
                signal = 0

        # Scale exposure by confidence and risk budget, capped by max leverage.
        weight = signal * min(self.risk.max_leverage, conf * self.risk.max_leverage)
        return RiskDecision(signal=signal, confidence=conf, target_weight=float(weight))

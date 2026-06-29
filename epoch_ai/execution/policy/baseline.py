"""Confidence-weighted ensemble baseline policy over multi-horizon forecasts.

This is the benchmark the learned RL policy must beat out-of-sample. It consumes
:class:`~epoch_ai.services.types.HorizonForecast` rows (not raw model logits) and
auto-drops heads that fail the reliability floor.
"""

from __future__ import annotations

from dataclasses import dataclass

from epoch_ai.services.types import DEFAULT_RELIABILITY_FLOOR, HorizonForecast


@dataclass(slots=True)
class BaselineDecision:
    """Direction + size hint from the confidence-weighted horizon ensemble."""

    signal: int
    confidence: float
    weighted_p_up: float
    n_heads_used: int
    skipped_horizons: tuple[int, ...]


def baseline_policy(
    forecasts: list[HorizonForecast],
    *,
    long_threshold: float = 0.55,
    short_threshold: float = 0.45,
    reliability_floor: float = DEFAULT_RELIABILITY_FLOOR,
) -> BaselineDecision:
    """Aggregate reliable horizons into a single directional view.

    Returns flat (``signal=0``) when no head passes the reliability floor or the
    weighted P(up) sits inside the dead band.
    """
    usable = [f for f in forecasts if f.reliable and f.confidence >= reliability_floor]
    skipped = tuple(sorted({f.horizon for f in forecasts if f.horizon not in {u.horizon for u in usable}}))
    if not usable:
        return BaselineDecision(0, 0.0, 0.5, 0, skipped)

    weight_sum = sum(f.confidence for f in usable)
    weighted_p = sum(f.confidence * f.p_up for f in usable) / weight_sum
    avg_conf = weight_sum / len(usable)

    if weighted_p >= long_threshold:
        signal = 1
    elif weighted_p <= short_threshold:
        signal = -1
    else:
        signal = 0

    return BaselineDecision(
        signal=signal,
        confidence=float(avg_conf),
        weighted_p_up=float(weighted_p),
        n_heads_used=len(usable),
        skipped_horizons=skipped,
    )

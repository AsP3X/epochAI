"""Pre-trade manipulation / rug-risk scoring (execution layer only)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from epoch_ai.config.settings import SafetyConfig

# Human: Weight map for combining soft risk indicators into a single suspicion score.
# Agent: READS manip_/oc_ feature snapshots; RETURNS SafetyAssessment in [0,1].
_INDICATOR_WEIGHTS: dict[str, float] = {
    "manip_illiq_spike": 0.25,
    "manip_wick_cluster": 0.15,
    "manip_vol_price_div": 0.15,
    "manip_return_kurt": 0.10,
    "manip_oi_price_div": 0.10,
    "manip_funding_extreme": 0.10,
    "manip_liq_spike": 0.15,
    "oc_liq_chg_z": 0.30,
    "oc_holder_conc": 0.20,
    "oc_lp_unlock_velocity": 0.25,
}


def _normalize_indicator(name: str, value: float) -> float:
    if name == "manip_illiq_spike":
        return min(1.0, max(0.0, value / 3.0))
    if name == "manip_wick_cluster":
        return min(1.0, max(0.0, value))
    if name == "manip_vol_price_div":
        return min(1.0, max(0.0, value / 3.0))
    if name == "manip_return_kurt":
        return min(1.0, max(0.0, abs(value) / 10.0))
    if name == "manip_oi_price_div":
        return min(1.0, max(0.0, value))
    if name == "manip_funding_extreme":
        return min(1.0, max(0.0, value / 3.0))
    if name == "manip_liq_spike":
        return min(1.0, max(0.0, value / 10.0))
    if name == "oc_liq_chg_z":
        return min(1.0, max(0.0, -value / 3.0))  # negative z = draining liquidity
    if name == "oc_holder_conc":
        return min(1.0, max(0.0, (value - 0.5) / 0.5))
    if name == "oc_lp_unlock_velocity":
        return min(1.0, max(0.0, value))
    return min(1.0, max(0.0, abs(value)))


@dataclass(slots=True)
class SafetyAssessment:
    """Combined pre-trade suspicion snapshot."""

    suspicion_score: float
    reasons: tuple[str, ...] = ()


class SafetyScorer:
    """Combine manipulation and on-chain feature snapshots into one score."""

    def __init__(self, config: SafetyConfig) -> None:
        self.config = config

    def assess(
        self,
        row: pd.Series | Mapping[str, float] | None = None,
        *,
        manip_features: Mapping[str, float] | None = None,
        onchain_features: Mapping[str, float] | None = None,
    ) -> SafetyAssessment:
        """Return suspicion in ``[0, 1]`` from feature snapshots."""
        if not self.config.enabled:
            return SafetyAssessment(suspicion_score=0.0)

        values: dict[str, float] = {}
        if row is not None:
            if isinstance(row, pd.Series):
                items = row.items()
            else:
                items = row.items()
            for key, val in items:
                if key.startswith("manip_") or key.startswith("oc_"):
                    values[key] = float(val)
        if manip_features:
            values.update({k: float(v) for k, v in manip_features.items()})
        if onchain_features:
            values.update({k: float(v) for k, v in onchain_features.items()})

        if self.config.block_on_missing_onchain and not any(k.startswith("oc_") for k in values):
            return SafetyAssessment(
                suspicion_score=1.0,
                reasons=("missing_onchain",),
            )

        weighted: list[tuple[float, str]] = []
        for name, weight in _INDICATOR_WEIGHTS.items():
            if name not in values:
                continue
            norm = _normalize_indicator(name, values[name])
            if norm <= 0.0:
                continue
            weighted.append((norm * weight, name))

        if not weighted:
            return SafetyAssessment(suspicion_score=0.0)

        score = min(1.0, max(w for w, _ in weighted))
        reasons = tuple(name for w, name in weighted if w >= score * 0.9)
        return SafetyAssessment(suspicion_score=score, reasons=reasons)

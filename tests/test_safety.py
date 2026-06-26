"""Tests for pre-trade safety scoring and risk integration."""

from __future__ import annotations

from epoch_ai.config.settings import PredictionConfig, RiskConfig, SafetyConfig
from epoch_ai.execution.risk import RiskManager
from epoch_ai.execution.safety import SafetyAssessment, SafetyScorer


def test_safety_scorer_bounds(market):
    scorer = SafetyScorer(SafetyConfig(enabled=True))
    row = market.iloc[-1]
    assessment = scorer.assess(row)
    assert 0.0 <= assessment.suspicion_score <= 1.0


def test_safety_scorer_detects_manipulation_features():
    scorer = SafetyScorer(SafetyConfig(enabled=True))
    assessment = scorer.assess({"manip_illiq_spike": 5.0, "manip_wick_cluster": 0.9})
    assert assessment.suspicion_score > 0.0
    assert assessment.reasons


def test_risk_manager_blocks_high_suspicion():
    risk = RiskConfig(long_threshold=0.5)
    mgr = RiskManager(
        risk,
        PredictionConfig(task="classification"),
        SafetyConfig(enabled=True, max_suspicion_score=0.5),
    )
    decision = mgr.decide(
        0.9,
        safety=SafetyAssessment(suspicion_score=0.9, reasons=("manip_illiq_spike",)),
    )
    assert decision.signal == 0
    assert decision.halted


def test_risk_manager_scales_weight_by_suspicion():
    risk = RiskConfig(long_threshold=0.5, risk_per_trade=0.02, max_leverage=3.0)
    mgr = RiskManager(
        risk,
        PredictionConfig(task="classification"),
        SafetyConfig(enabled=True, max_suspicion_score=0.99, scale_weight_by_suspicion=True),
    )
    base = mgr.decide(0.9)
    scaled = mgr.decide(
        0.9,
        safety=SafetyAssessment(suspicion_score=0.5, reasons=("manip_wick_cluster",)),
    )
    assert abs(scaled.target_weight) < abs(base.target_weight)

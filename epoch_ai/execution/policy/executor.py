"""Bridge multi-horizon forecasts to :class:`RiskDecision` via baseline or PPO."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from epoch_ai.config.settings import AppConfig
from epoch_ai.execution.policy.baseline import baseline_policy
from epoch_ai.execution.policy.guardrails import apply_guardrails
from epoch_ai.execution.policy.observation import build_runtime_observation
from epoch_ai.execution.policy.ppo_policy import PPOPolicy
from epoch_ai.execution.portfolio_state import PortfolioState
from epoch_ai.execution.risk import RiskDecision, RiskManager
from epoch_ai.execution.safety import SafetyAssessment
from epoch_ai.services.types import MultiHorizonPredictionResult


def load_ppo_policy(config: AppConfig) -> PPOPolicy | None:
    """Load the learned policy when configured and the artifact exists."""
    path = Path(config.rl.policy_path)
    if not path.exists():
        return None
    return PPOPolicy.load(path, config.rl)


def baseline_weight(
    config: AppConfig,
    multi: MultiHorizonPredictionResult,
    portfolio: PortfolioState,
) -> float:
    """Size a baseline-ensemble target weight within risk caps (no guardrails)."""
    long_t = config.risk.long_threshold
    short_t = config.risk.short_threshold
    if config.trading.trade_frequency == "active":
        long_t = min(long_t, 0.52)
        short_t = max(short_t, 0.48)
    decision = baseline_policy(
        multi.horizons,
        long_threshold=long_t,
        short_threshold=short_t,
        reliability_floor=config.trading.reliability_floor,
    )
    if decision.signal == 0:
        return 0.0
    cap = config.trading.max_position_fraction * config.risk.max_leverage
    size = min(
        cap,
        decision.confidence * config.risk.risk_per_trade * config.risk.max_leverage,
    )
    return float(decision.signal * size)


def decide_trading_action(
    config: AppConfig,
    *,
    raw_prediction: float,
    multi: MultiHorizonPredictionResult | None,
    portfolio: PortfolioState,
    ppo: PPOPolicy | None = None,
    safety: SafetyAssessment | None = None,
    trunk_embedding: np.ndarray | None = None,
) -> RiskDecision:
    """Choose a sized position using threshold, baseline, or learned policy."""
    backend = config.trading.policy_backend
    if backend == "threshold" or multi is None:
        return RiskManager(config.risk, config.prediction, config.safety).decide(
            raw_prediction,
            portfolio,
            safety=safety,
        )

    target = 0.0
    conf = 0.0
    if backend in {"baseline", "learned_with_baseline_fallback"}:
        target = baseline_weight(config, multi, portfolio)
        conf = min(1.0, abs(target) / max(1e-9, config.risk.max_leverage))

    if backend in {"learned", "learned_with_baseline_fallback"}:
        policy = ppo or load_ppo_policy(config)
        if policy is not None:
            obs = build_runtime_observation(
                config, multi, portfolio, trunk_embedding=trunk_embedding
            )
            learned = policy.act(obs, deterministic=True)
            cap = config.trading.max_position_fraction * config.risk.max_leverage
            target = float(max(-cap, min(cap, learned * cap)))
            conf = min(1.0, abs(target) / max(1e-9, cap))
        elif backend == "learned":
            target = 0.0

    if safety is not None and config.safety.enabled:
        if safety.suspicion_score >= config.safety.max_suspicion_score:
            return RiskDecision(0, conf, 0.0, halted=True)

    target = apply_guardrails(target, portfolio, config.trading, config.risk)
    signal = 0 if abs(target) < 1e-9 else (1 if target > 0 else -1)
    if not config.risk.allow_short and signal < 0:
        target = 0.0
        signal = 0
    return RiskDecision(
        signal=signal,
        confidence=conf,
        target_weight=target,
        halted=False,
    )

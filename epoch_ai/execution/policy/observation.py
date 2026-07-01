"""Build fixed-size policy observation vectors from forecasts + portfolio state."""

from __future__ import annotations

import numpy as np

from epoch_ai.config.settings import AppConfig
from epoch_ai.execution.portfolio_state import PortfolioState
from epoch_ai.services.types import HorizonForecast, MultiHorizonPredictionResult


def decision_horizons(config: AppConfig) -> list[int]:
    """Horizons included in the policy observation."""
    if config.trading.decision_horizons:
        return list(config.trading.decision_horizons)
    return list(config.prediction.horizons)


def observation_dim(config: AppConfig) -> int:
    """Flat observation size: 3 features per horizon + 4 portfolio scalars."""
    return len(decision_horizons(config)) * 3 + 4


def embedding_observation_dim(trunk_dim: int) -> int:
    """Flat embedding observation size: trunk embedding + 4 portfolio scalars."""
    return trunk_dim + 4


def build_embedding_observation(
    embedding_row: np.ndarray,
    portfolio: PortfolioState,
    config: AppConfig,
) -> np.ndarray:
    """Concatenate the trunk embedding with the same 4 portfolio scalars as build_observation."""
    # Agent: same portfolio scalars/order as build_observation so the two obs modes are
    #        interchangeable except for the leading feature block (embedding vs forecasts).
    parts = list(np.asarray(embedding_row, dtype=float).ravel())
    parts.extend(
        [
            portfolio.position_weight,
            portfolio.drawdown(),
            portfolio.session_loss(),
            float(portfolio.bars_in_position) / max(1, config.trading.max_hold_bars),
        ]
    )
    return np.asarray(parts, dtype=np.float32)


def policy_env_observation(
    env,
    config: AppConfig,
) -> np.ndarray:
    """Build the policy observation vector for a replay env at its current bar.

    Matches the training path: embedding mode uses the trunk row at ``env._pos`` when
    ``env.embeddings`` is set; otherwise the per-horizon forecast summary.
    """
    if config.rl.observation_mode == "embedding" and env.embeddings is not None:
        return build_embedding_observation(env.embeddings[env._pos], env.portfolio, config)
    return build_observation(env.current_forecast(), env.portfolio, config)


def build_runtime_observation(
    config: AppConfig,
    multi: MultiHorizonPredictionResult | None,
    portfolio: PortfolioState,
    *,
    trunk_embedding: np.ndarray | None = None,
) -> np.ndarray:
    """Build the policy observation for live/replay (forecast or embedding mode)."""
    if config.rl.observation_mode == "embedding" and trunk_embedding is not None:
        return build_embedding_observation(trunk_embedding, portfolio, config)
    return build_observation(multi, portfolio, config)


def build_observation(
    multi: MultiHorizonPredictionResult | None,
    portfolio: PortfolioState,
    config: AppConfig,
) -> np.ndarray:
    """Concatenate reliable per-horizon features and portfolio context."""
    horizons = decision_horizons(config)
    floor = config.trading.reliability_floor
    by_h: dict[int, HorizonForecast] = {}
    if multi is not None:
        by_h = {f.horizon: f for f in multi.horizons}

    parts: list[float] = []
    for h in horizons:
        forecast = by_h.get(h)
        if forecast is None or not forecast.reliable or forecast.confidence < floor:
            parts.extend([0.5, 0.0, 0.0])
        else:
            parts.extend([forecast.p_up, forecast.exp_return, forecast.confidence])

    parts.extend(
        [
            portfolio.position_weight,
            portfolio.drawdown(),
            portfolio.session_loss(),
            float(portfolio.bars_in_position) / max(1, config.trading.max_hold_bars),
        ]
    )
    return np.asarray(parts, dtype=np.float32)

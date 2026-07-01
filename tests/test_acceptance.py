"""Tests for holdout acceptance helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from epoch_ai.learning.acceptance import holdout_replay_env_factories
from tests.test_policy_env import _policy_config


def test_holdout_env_factories_use_forecast_for_baseline_in_embedding_mode():
    config = _policy_config(rl={"observation_mode": "embedding"})
    market = pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0]})
    structured = {
        4: {
            "p_up": np.array([0.5, 0.51, 0.52, 0.53], dtype=float),
            "q10": np.zeros(4),
            "q50": np.zeros(4),
            "q90": np.zeros(4),
        }
    }
    tcn = MagicMock(name="tcn")

    with patch(
        "epoch_ai.execution.policy.trunk_policy.build_embedding_env"
    ) as mock_build:
        mock_build.return_value = MagicMock(embeddings=np.zeros((4, 8)))
        baseline_factory, policy_factory = holdout_replay_env_factories(
            config,
            market,
            structured=structured,
            real_horizons=[4],
            data_index=market.index,
            tcn_model=tcn,
        )

        baseline_env = baseline_factory()
        assert baseline_env.structured_forecasts is not None
        assert baseline_env.embeddings is None

        policy_factory()
        mock_build.assert_called_once_with(config, market, tcn)


def test_holdout_env_factories_share_forecast_env_when_not_embedding():
    config = _policy_config(rl={"observation_mode": "forecast"})
    market = pd.DataFrame({"close": [100.0, 101.0, 102.0]})
    structured = {
        4: {
            "p_up": np.array([0.5, 0.51, 0.52], dtype=float),
            "q10": np.zeros(3),
            "q50": np.zeros(3),
            "q90": np.zeros(3),
        }
    }

    baseline_factory, policy_factory = holdout_replay_env_factories(
        config,
        market,
        structured=structured,
        real_horizons=[4],
        data_index=market.index,
        tcn_model=None,
    )

    baseline_env = baseline_factory()
    policy_env = policy_factory()
    assert baseline_env.structured_forecasts is not None
    assert policy_env.structured_forecasts is not None

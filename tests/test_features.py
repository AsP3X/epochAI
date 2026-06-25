"""Tests for the feature pipeline and target construction."""

from __future__ import annotations

import numpy as np

from epoch_ai.features.pipeline import FeaturePipeline, build_target, forward_return


def test_pipeline_produces_features(market, small_config):
    features = FeaturePipeline(small_config).transform(market)
    assert features.shape[0] > 0
    assert features.shape[1] > 20
    assert not features.isna().any().any()  # dropna=True by default


def test_feature_groups_present(market, small_config):
    features = FeaturePipeline(small_config).transform(market)
    prefixes = {name.split("_")[0] for name in features.columns}
    for expected in ["ta", "micro", "deriv", "vol", "time"]:
        assert expected in prefixes


def test_target_is_binary_and_causal(market, small_config):
    target = build_target(market, small_config.prediction)
    valid = target.dropna()
    assert set(np.unique(valid.to_numpy())).issubset({0.0, 1.0})
    # Final `horizon` rows have no realised future.
    assert target.iloc[-small_config.prediction.horizon :].isna().all()


def test_forward_return_alignment(market, small_config):
    fr = forward_return(market, small_config.prediction.horizon)
    assert fr.iloc[-small_config.prediction.horizon :].isna().all()

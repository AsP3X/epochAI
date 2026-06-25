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


def test_new_technical_indicators_present(market, small_config):
    features = FeaturePipeline(small_config).transform(market)
    for col in ["ta_adx_14", "ta_williams_r", "ta_cci_20", "ta_vwap_dist", "ta_obv_z"]:
        assert col in features.columns


def test_config_driven_windows(market, small_config):
    """Changing window config changes the emitted feature columns."""
    small_config.features.ma_windows = [5, 15]
    small_config.features.rsi_periods = [10]
    features = FeaturePipeline(small_config).transform(market)
    assert "ta_sma_dist_5" in features.columns
    assert "ta_sma_dist_15" in features.columns
    assert "ta_rsi_10" in features.columns
    # The default windows should no longer be present.
    assert "ta_sma_dist_200" not in features.columns
    assert "ta_rsi_14" not in features.columns


def test_onchain_group_activates_with_columns(market, small_config):
    """Enabling onchain wires the group; it emits columns when data is present."""
    small_config.features.onchain = True
    enriched = market.copy()
    enriched["exchange_netflow"] = np.linspace(-1.0, 1.0, len(enriched))
    enriched["active_addresses"] = np.linspace(100.0, 200.0, len(enriched))
    features = FeaturePipeline(small_config).transform(enriched)
    assert "oc_netflow_z" in features.columns
    assert "oc_active_dist" in features.columns


def test_sentiment_group_uses_fear_greed(market, small_config):
    small_config.features.sentiment = True
    enriched = market.copy()
    rng = np.random.default_rng(1)
    enriched["fear_greed"] = rng.uniform(0, 100, len(enriched))
    features = FeaturePipeline(small_config).transform(enriched)
    assert "sent_fear_greed" in features.columns
    assert "sent_fear_greed_z" in features.columns

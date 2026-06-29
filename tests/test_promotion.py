"""Tests for the automated retrain -> evaluate -> promote-if-better pipeline."""

from __future__ import annotations

import math

import pytest

from epoch_ai.features.pipeline import FeaturePipeline, build_target
from epoch_ai.learning.promotion import (
    auto_retrain_and_promote,
    decide_promotion,
    metric_higher_is_better,
)
from epoch_ai.models.lightgbm_model import LightGBMModel
from epoch_ai.models.registry import ModelRegistry


def test_metric_direction():
    assert metric_higher_is_better("oos_accuracy") is True
    assert metric_higher_is_better("oos_auc") is True
    assert metric_higher_is_better("oos_logloss") is False
    assert metric_higher_is_better("oos_brier") is False
    assert metric_higher_is_better("oos_rmse") is False


def test_decide_promotion_directions_and_threshold():
    # Lower-is-better (logloss): a smaller challenger value wins.
    assert decide_promotion(0.70, 0.60, metric="oos_logloss", min_improvement=0.0)[0] is True
    assert decide_promotion(0.60, 0.70, metric="oos_logloss", min_improvement=0.0)[0] is False
    # Higher-is-better (accuracy): a larger challenger value wins.
    assert decide_promotion(0.50, 0.60, metric="oos_accuracy", min_improvement=0.0)[0] is True
    assert decide_promotion(0.60, 0.50, metric="oos_accuracy", min_improvement=0.0)[0] is False
    # Margin must clear min_improvement.
    assert decide_promotion(0.600, 0.595, metric="oos_logloss", min_improvement=0.01)[0] is False
    assert decide_promotion(0.600, 0.580, metric="oos_logloss", min_improvement=0.01)[0] is True


def test_decide_promotion_tie_never_promotes():
    # A tie (identical metric) must NOT promote, even with the default 0 floor:
    # promoting an identically-scoring clone only churns the registry/champion pointer.
    promote, reason = decide_promotion(
        0.693106, 0.693106, metric="oos_logloss", min_improvement=0.0
    )
    assert promote is False
    assert "<= 0.000000" in reason
    # Same for a higher-is-better metric tie.
    assert decide_promotion(0.55, 0.55, metric="oos_accuracy", min_improvement=0.0)[0] is False
    # A strictly positive improvement still promotes with the 0 floor.
    assert decide_promotion(0.693106, 0.693105, metric="oos_logloss", min_improvement=0.0)[0] is True


def test_decide_promotion_bootstrap_and_nan():
    # No champion -> bootstrap promote.
    assert decide_promotion(None, 0.6, metric="oos_logloss", min_improvement=0.0)[0] is True
    # Undefined challenger metric -> never promote.
    assert decide_promotion(0.6, math.nan, metric="oos_logloss", min_improvement=0.0)[0] is False
    # Undefined champion metric -> promote the (valid) challenger.
    assert decide_promotion(math.nan, 0.6, metric="oos_logloss", min_improvement=0.0)[0] is True


def test_registry_promotion_pointer(market, small_config, tmp_path):
    small_config.model.model_dir = str(tmp_path / "models")
    registry = ModelRegistry(small_config.model.model_dir)

    feats = FeaturePipeline(small_config).transform(market)
    y = build_target(market, small_config.prediction)
    data = feats.join(y).dropna()
    x, yy = data[feats.columns], data["target"]

    v1 = registry.save(LightGBMModel(small_config.model).fit(x.iloc[:1000], yy.iloc[:1000]))
    v2 = registry.save(LightGBMModel(small_config.model).fit(x.iloc[:1200], yy.iloc[:1200]))

    # No pointer yet => default resolves to the latest version (legacy behaviour).
    assert registry.latest_label() == v2
    assert registry.promoted_label() is None
    _, meta = registry.load(None, small_config.model)
    assert meta["label"] == v2

    # Promote the older version; default resolution now follows the pointer.
    registry.set_promoted(v1, info={"metric": "oos_logloss", "value": 0.69})
    assert registry.promoted_label() == v1
    _, meta = registry.load(None, small_config.model)
    assert meta["label"] == v1

    # An explicit label still overrides the pointer.
    _, meta = registry.load(v2, small_config.model)
    assert meta["label"] == v2

    # Promoting an unknown version is rejected.
    with pytest.raises(FileNotFoundError):
        registry.set_promoted("v_999")


@pytest.mark.slow
def test_auto_retrain_bootstrap_promotes(small_config, tmp_path):
    small_config.model.model_dir = str(tmp_path / "models")
    small_config.data.data_dir = str(tmp_path / "data")
    small_config.promotion.eval_bars = 800

    result = auto_retrain_and_promote(small_config, n_bars=4000)

    assert result.skipped is False
    assert result.champion_label is None      # empty registry -> bootstrap
    assert result.promoted is True
    assert result.challenger_label is not None

    registry = ModelRegistry(small_config.model.model_dir)
    assert registry.promoted_label() == result.challenger_label
    model, meta = registry.load(None, small_config.model)
    assert meta["label"] == result.challenger_label
    assert model.feature_names_


@pytest.mark.slow
def test_auto_retrain_second_cycle_keeps_loadable_champion(small_config, tmp_path):
    small_config.model.model_dir = str(tmp_path / "models")
    small_config.data.data_dir = str(tmp_path / "data")
    small_config.promotion.eval_bars = 800

    first = auto_retrain_and_promote(small_config, n_bars=4000)
    second = auto_retrain_and_promote(small_config, n_bars=4000)

    # The second cycle's incumbent is the first cycle's promoted challenger.
    assert second.champion_label == first.challenger_label
    assert second.skipped is False

    # Whatever is promoted must always be a loadable version.
    registry = ModelRegistry(small_config.model.model_dir)
    promoted = registry.promoted_label()
    assert promoted in {first.challenger_label, second.challenger_label}
    model, meta = registry.load(None, small_config.model)
    assert meta["label"] == promoted
    assert model.feature_names_


def test_auto_retrain_skips_without_enough_data(small_config, tmp_path):
    small_config.model.model_dir = str(tmp_path / "models")
    small_config.data.data_dir = str(tmp_path / "data")
    small_config.promotion.eval_bars = 5000  # more than the available history

    # Too few bars to leave initial_train_period rows behind the embargo + holdout.
    result = auto_retrain_and_promote(small_config, n_bars=950)
    assert result.skipped is True
    assert result.promoted is False
    assert ModelRegistry(small_config.model.model_dir).promoted_label() is None

"""Tests for the LightGBM model wrapper."""

from __future__ import annotations

import numpy as np

from epoch_ai.features.pipeline import FeaturePipeline, build_target
from epoch_ai.models.lightgbm_model import LightGBMModel


def _xy(market, config):
    features = FeaturePipeline(config).transform(market)
    target = build_target(market, config.prediction)
    data = features.join(target).dropna()
    return data[features.columns], data["target"]


def test_fit_predict_save_load(market, small_config, tmp_path):
    x, y = _xy(market, small_config)
    model = LightGBMModel(small_config.model, task="classification")
    model.fit(x.iloc[:1500], y.iloc[:1500])

    preds = model.predict(x.iloc[1500:1600])
    assert preds.shape[0] == 100
    assert ((preds >= 0) & (preds <= 1)).all()

    path = tmp_path / "model.txt"
    model.save(str(path))
    loaded = LightGBMModel.load(str(path), small_config.model)
    reloaded = loaded.predict(x.iloc[1500:1600])
    assert np.allclose(preds, reloaded, atol=1e-9)


def test_feature_importance(market, small_config):
    x, y = _xy(market, small_config)
    model = LightGBMModel(small_config.model).fit(x.iloc[:1500], y.iloc[:1500])
    importance = model.feature_importance()
    assert len(importance) == x.shape[1]
    assert (importance >= 0).all()

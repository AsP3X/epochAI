"""Multi-head evolved_nn layout and training tests."""

from __future__ import annotations

import numpy as np
import pytest

from epoch_ai.features.pipeline import build_multi_horizon_targets
from epoch_ai.models.multi_head import MultiHeadSpec, targets_to_matrix

pytest.importorskip("torch")


def test_targets_to_matrix_shape(market, small_config):
    small_config.prediction.horizons = [1, 5]
    small_config.prediction.horizon = 5
    spec = MultiHeadSpec.from_prediction(small_config.prediction)
    targets = build_multi_horizon_targets(market, small_config.prediction)
    mat = targets_to_matrix(targets, spec)
    assert mat.shape == (len(market), spec.n_outputs)


def test_multi_head_fast_fit_and_predict(market, small_config, tmp_path):
    from epoch_ai.models.evolved_nn_model import EvolvedNNModel

    small_config.prediction.horizons = [1, 5]
    small_config.prediction.horizon = 5
    small_config.prediction.quantiles = [0.1, 0.5, 0.9]
    small_config.model.evolution.fast_fit = True
    small_config.model.nn.max_epochs = 5
    small_config.model.nn.patience = 2
    small_config.model.calibration = "none"

    from epoch_ai.features.pipeline import FeaturePipeline

    features = FeaturePipeline(small_config).transform(market)
    multi = build_multi_horizon_targets(market, small_config.prediction)
    aligned = features.join(multi).dropna()
    x = aligned[features.columns]
    y = aligned["target_5"]
    mt = aligned[[c for c in aligned.columns if c.startswith(("ret_", "target_"))]]

    model = EvolvedNNModel(small_config.model, task="classification")
    model.fit(
        x.iloc[:800],
        y.iloc[:800],
        prediction=small_config.prediction,
        multi_targets=mt.iloc[:800],
    )
    assert model.multi_head_spec_ is not None
    assert model.multi_head_spec_.n_outputs == 8  # 2 horizons x 4 outputs

    probs = model.predict(x.iloc[800:810])
    assert probs.shape == (10,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))

    structured = model.predict_structured(x.iloc[800:805])
    assert 1 in structured and 5 in structured
    assert "p_up" in structured[5]
    assert "q50" in structured[5]

    path = tmp_path / "model.pt"
    model.save(str(path))
    loaded = EvolvedNNModel.load(str(path), small_config.model, task="classification")
    assert loaded.multi_head_spec_ is not None
    assert loaded.multi_head_spec_.n_outputs == 8
    reloaded = loaded.predict(x.iloc[800:810])
    np.testing.assert_allclose(reloaded, probs, rtol=0, atol=1e-5)


def test_multi_head_isotonic_calibration_roundtrip(market, small_config, tmp_path):
    from epoch_ai.features.pipeline import FeaturePipeline
    from epoch_ai.models.evolved_nn_model import EvolvedNNModel

    small_config.prediction.horizons = [1, 5]
    small_config.prediction.horizon = 5
    small_config.prediction.quantiles = [0.1, 0.5, 0.9]
    small_config.model.evolution.fast_fit = True
    small_config.model.nn.max_epochs = 5
    small_config.model.nn.patience = 2
    small_config.model.calibration = "isotonic"

    features = FeaturePipeline(small_config).transform(market)
    multi = build_multi_horizon_targets(market, small_config.prediction)
    aligned = features.join(multi).dropna()
    x = aligned[features.columns]
    y = aligned["target_5"]
    mt = aligned[[c for c in aligned.columns if c.startswith(("ret_", "target_"))]]

    model = EvolvedNNModel(small_config.model, task="classification")
    model.fit(
        x.iloc[:800],
        y.iloc[:800],
        prediction=small_config.prediction,
        multi_targets=mt.iloc[:800],
    )
    assert model.multi_calibrator_ is not None

    path = tmp_path / "model.pt"
    model.save(str(path))
    loaded = EvolvedNNModel.load(str(path), small_config.model, task="classification")
    assert loaded.multi_calibrator_ is not None
    np.testing.assert_allclose(
        loaded.predict(x.iloc[800:810]),
        model.predict(x.iloc[800:810]),
        rtol=0,
        atol=1e-5,
    )
    structured = loaded.predict_structured(x.iloc[800:805])
    assert 1 in structured and 5 in structured

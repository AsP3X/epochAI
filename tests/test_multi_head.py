"""Multi-head evolved_nn layout and training tests."""

from __future__ import annotations

import numpy as np
import pytest

from epoch_ai.features.pipeline import build_multi_horizon_targets
from epoch_ai.models.multi_head import (
    MultiHeadSpec,
    _stable_sigmoid,
    multi_head_val_loss,
    targets_to_matrix,
)

pytest.importorskip("torch")


def test_stable_sigmoid_extreme_logits_no_overflow():
    logits = np.array([-1000.0, 1000.0, 0.0])
    with np.errstate(over="raise"):
        probs = _stable_sigmoid(logits)
    np.testing.assert_allclose(probs, [0.0, 1.0, 0.5], rtol=0, atol=1e-6)


def test_multi_head_val_loss_extreme_logits(small_config):
    small_config.prediction.horizons = [1, 5]
    small_config.prediction.horizon = 5
    spec = MultiHeadSpec.from_prediction(small_config.prediction)
    n = 32
    logits = np.zeros((n, spec.n_outputs), dtype=np.float64)
    y = np.zeros((n, spec.n_outputs), dtype=np.float64)
    dir_idx = spec.direction_index(5)
    logits[:, dir_idx] = np.linspace(-500.0, 500.0, n)
    y[:, dir_idx] = np.random.default_rng(0).integers(0, 2, size=n)
    with np.errstate(over="raise"):
        loss = multi_head_val_loss(logits, y, spec, primary_horizon=5)
    assert np.isfinite(loss)


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


def test_multi_head_val_loss_torch_finite_and_close(small_config):
    """GPU val loss stays on device and tracks the numpy metric for early stopping."""
    import torch

    from epoch_ai.models.multi_head import multi_head_val_loss_torch

    small_config.prediction.horizons = [1, 5, 10]
    spec = MultiHeadSpec.from_prediction(small_config.prediction)
    rng = np.random.default_rng(0)
    n = 64
    logits_np = rng.standard_normal((n, spec.n_outputs)).astype(np.float32) * 0.1
    y_np = rng.integers(0, 2, size=(n, spec.n_outputs)).astype(np.float32)
    for h in spec.horizons:
        base = spec.head_offset(h)
        for k in range(len(spec.quantiles)):
            y_np[:, base + k] = rng.standard_normal(n).astype(np.float32) * 0.01

    np_loss = multi_head_val_loss(logits_np, y_np, spec, primary_horizon=10)
    torch_loss = multi_head_val_loss_torch(
        torch.from_numpy(logits_np),
        torch.from_numpy(y_np),
        spec,
        primary_horizon=10,
    )
    assert np.isfinite(torch_loss)
    assert abs(torch_loss - np_loss) < 0.15

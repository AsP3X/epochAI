"""Tests for training and runtime service layer."""

from __future__ import annotations

import json

import pytest

from epoch_ai.data.downloader import HistoricalDownloader
from epoch_ai.logging_system.joiner import retrain_log_stats
from epoch_ai.logging_system.store import PredictionStore
from epoch_ai.models.registry import ModelRegistry
from epoch_ai.services.runtime import RuntimeService
from epoch_ai.services.training import TrainingService

pytestmark = pytest.mark.slow


def test_training_service_train(small_config, tmp_path):
    small_config.model.model_dir = str(tmp_path / "models")
    small_config.data.data_dir = str(tmp_path / "data")
    service = TrainingService(small_config)
    result = service.train(n_bars=2500, max_steps=2, register=True)
    assert result.model_version is not None
    assert result.walk_forward_steps >= 1
    assert len(service.list_models()) >= 1


def test_runtime_service_predict_and_run(market, small_config, tmp_path):
    small_config.model.model_dir = str(tmp_path / "models")
    small_config.data.data_dir = str(tmp_path / "data")

    train = TrainingService(small_config)
    train.train(n_bars=2500, max_steps=2, register=True)

    runtime = RuntimeService(small_config)
    pred = runtime.predict_market(market)
    assert 0.0 <= pred.raw_prediction <= 1.0
    assert pred.model_version.startswith("v_")

    multi = runtime.predict_multi_horizon(market)
    assert multi.last_close > 0
    assert len(multi.horizons) >= 1
    assert multi.to_json()["model_version"].startswith("v_")

    result = runtime.run_session(n_bars=2500, live_bars=200, retrain_every=0)
    assert result.bars_processed > 0


def test_runtime_session_log_predictions(market, small_config, tmp_path, monkeypatch):
    small_config.model.model_dir = str(tmp_path / "models")
    small_config.data.data_dir = str(tmp_path / "data")
    small_config.data.use_synthetic_fallback = True
    small_config.data.context_symbols = []
    small_config.data.fetch_fear_greed = False
    small_config.features.cross_asset = False
    small_config.features.sentiment = False
    small_config.logging.db_path = str(tmp_path / "logs" / "predictions.sqlite")

    def fake_load(
        self,
        symbol=None,
        *,
        n_bars=None,
        align_index=None,
        force=False,
        fetch_if_missing=True,
        skip_enrichment=False,
    ):
        del symbol, align_index, force, fetch_if_missing, skip_enrichment
        cap = len(market) if n_bars is None else min(n_bars, len(market))
        return market.iloc[:cap].copy()

    monkeypatch.setattr(HistoricalDownloader, "load_or_download", fake_load)

    TrainingService(small_config).train(n_bars=2500, max_steps=2, register=True)
    runtime = RuntimeService(small_config)
    runtime.run_session(n_bars=2500, live_bars=200, log_predictions=True)

    store = PredictionStore(small_config.logging.db_path)
    try:
        stats = retrain_log_stats(store, small_config.primary_symbol)
    finally:
        store.close()
    assert stats.predictions == 200
    assert stats.joined_samples == 200 - small_config.prediction.horizon
    assert stats.pending == small_config.prediction.horizon


def test_runtime_requires_trained_model(small_config, tmp_path):
    small_config.model.model_dir = str(tmp_path / "models")
    runtime = RuntimeService(small_config)
    assert runtime.status().models_available == 0
    with pytest.raises(FileNotFoundError):
        runtime.load_model()


def test_registry_load(small_config, tmp_path):
    small_config.model.model_dir = str(tmp_path / "models")
    small_config.data.data_dir = str(tmp_path / "data")
    TrainingService(small_config).train(n_bars=2500, max_steps=1, register=True)

    registry = ModelRegistry(small_config.model.model_dir)
    versions = registry.list_versions()
    assert len(versions) == 1
    model, meta = registry.load(None, small_config.model, task="classification")
    assert meta["label"] == versions[0]["label"]
    assert meta.get("open_weights") is True
    assert model.feature_names_ is not None


def test_registry_export_open_bundle(small_config, tmp_path):
    small_config.model.model_dir = str(tmp_path / "models")
    small_config.data.data_dir = str(tmp_path / "data")
    TrainingService(small_config).train(n_bars=2500, max_steps=1, register=True)

    registry = ModelRegistry(small_config.model.model_dir)
    bundle = registry.export_open_bundle(tmp_path / "release")
    assert (bundle / "model.txt").exists()
    assert (bundle / "metadata.json").exists()
    assert (bundle / "README.txt").exists()
    meta = json.loads((bundle / "metadata.json").read_text())
    assert meta["open_weights"] is True

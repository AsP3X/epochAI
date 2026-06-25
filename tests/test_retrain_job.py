"""Tests for scheduled retrain job."""

from __future__ import annotations

from epoch_ai.learning.retrain_job import run_retrain


def test_retrain_from_parquet_fallback(small_config, tmp_path, monkeypatch):
    small_config.model.model_dir = str(tmp_path / "models")
    small_config.data.data_dir = str(tmp_path / "data")
    small_config.logging.db_path = str(tmp_path / "empty.sqlite")

    result = run_retrain(small_config, min_new_samples=999, register=True, n_bars=2000)
    assert not result.skipped
    assert result.train_rows > 0
    assert result.model_version is not None

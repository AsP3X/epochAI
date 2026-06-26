"""Tests for TrainingService real-data policy."""

from __future__ import annotations

from epoch_ai.config.settings import AppConfig
from epoch_ai.services.training import TrainingService


def test_evolved_nn_disables_synthetic_fallback_in_download_config():
    cfg = AppConfig.model_validate(
        {
            "data": {"use_synthetic_fallback": True},
            "model": {"backend": "evolved_nn"},
        }
    )
    resolved = TrainingService(cfg)._training_data_config()
    assert resolved.data.use_synthetic_fallback is False


def test_lightgbm_keeps_synthetic_fallback_setting():
    cfg = AppConfig.model_validate(
        {
            "data": {"use_synthetic_fallback": True},
            "model": {"backend": "lightgbm"},
        }
    )
    resolved = TrainingService(cfg)._training_data_config()
    assert resolved.data.use_synthetic_fallback is True

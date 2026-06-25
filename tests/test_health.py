"""Tests for live health checks."""

from __future__ import annotations

from epoch_ai.monitoring.health import check_live_health


def test_health_no_models(small_config, tmp_path):
    small_config.model.model_dir = str(tmp_path / "models")
    health = check_live_health(small_config)
    assert not health.ready
    assert "No trained models" in health.issues[0]

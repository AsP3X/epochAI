"""Tests for open-weights export."""

from __future__ import annotations

from epoch_ai.export.model_card import export_bundle_with_card
from epoch_ai.services.training import TrainingService


def test_export_bundle_with_model_card(small_config, tmp_path):
    small_config.model.model_dir = str(tmp_path / "models")
    small_config.data.data_dir = str(tmp_path / "data")
    TrainingService(small_config).train(n_bars=2500, max_steps=2, register=True)
    bundle = export_bundle_with_card(small_config, dest=str(tmp_path / "exports"))
    assert (bundle / "model.txt").exists()
    assert (bundle / "metadata.json").exists()
    assert (bundle / "MODEL_CARD.md").exists()
    card = (bundle / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert "epochAI Model Card" in card

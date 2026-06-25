"""Open-weights export helpers and model card generation."""

from __future__ import annotations

from pathlib import Path

from epoch_ai.config.settings import AppConfig
from epoch_ai.models.registry import ModelRegistry


def render_model_card(config: AppConfig, metadata: dict, label: str) -> str:
    """Render a plain-text model card for an exported bundle."""
    metrics = metadata.get("metrics", {})
    backend = metadata.get("backend", "lightgbm")
    weights_desc = (
        "plain XGBoost JSON weights" if backend == "xgboost" else "plain LightGBM weights"
    )
    lines = [
        "# epochAI Model Card",
        "",
        f"**Version:** {label}",
        f"**Symbol:** {metadata.get('symbol', config.primary_symbol)}",
        f"**Timeframe:** {config.timeframe}",
        f"**Task:** {metadata.get('task', config.prediction.task)}",
        f"**Backend:** {backend}",
        f"**Horizon:** {config.prediction.horizon} candles",
        f"**Features:** {metadata.get('n_features', 'unknown')}",
        f"**Created:** {metadata.get('created_at', 'unknown')}",
        "",
        "## Open weights",
        "",
        f"This bundle contains {weights_desc} "
        f"(`{metadata.get('model_file', 'model.txt')}`) and JSON metadata.",
        "No license is bundled — see repository owner for terms.",
        "",
        "## Training metrics",
        "",
    ]
    if metrics:
        for key, value in sorted(metrics.items()):
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- (no metrics recorded in metadata)")
    lines.extend(
        [
            "",
            "## Usage",
            "",
            "```python",
            "from epoch_ai.config.settings import AppConfig",
            "from epoch_ai.models.factory import model_class",
            "",
            "cfg = AppConfig()",
            f'model = model_class("{backend}").load(',
            f'    "{metadata.get("model_file", "model.txt")}", cfg.model,'
            f' task="{config.prediction.task}")',
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def export_bundle_with_card(
    config: AppConfig,
    *,
    dest: str | Path = "artifacts/exports",
    label: str | None = None,
) -> Path:
    """Export an open-weights bundle and write a MODEL_CARD.md alongside it."""
    registry = ModelRegistry(config.model.model_dir)
    resolved = label or registry.latest_label()
    if not resolved:
        raise FileNotFoundError("No models to export.")

    _, meta = registry.load(resolved, config.model, task=config.prediction.task)
    bundle_root = registry.export_open_bundle(dest, label=resolved)
    card_path = bundle_root / "MODEL_CARD.md"
    card_path.write_text(render_model_card(config, meta, resolved), encoding="utf-8")
    return bundle_root

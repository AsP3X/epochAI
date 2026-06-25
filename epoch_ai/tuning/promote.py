"""Promote the best experiment from a tune sweep."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from epoch_ai.config.overrides import apply_overrides
from epoch_ai.config.settings import AppConfig
from epoch_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class PromoteResult:
    """Outcome of selecting and applying the best sweep experiment."""

    experiment: str
    metric: str
    metric_value: float
    overrides: dict[str, Any]
    promoted_config_path: Path | None = None


def find_best_experiment(
    sweep_out: str | Path,
    *,
    metric: str = "sharpe",
) -> tuple[str, float, dict[str, Any]]:
    """Return the best experiment name, metric value, and its overrides."""
    root = Path(sweep_out)
    if not root.exists():
        raise FileNotFoundError(f"Sweep output not found: {root}")

    best_name = ""
    best_value = float("-inf")
    best_overrides: dict[str, Any] = {}

    for exp_dir in sorted(root.iterdir()):
        if not exp_dir.is_dir():
            continue
        metrics_path = exp_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        value = float(payload.get("strategy", {}).get(metric, float("-inf")))
        if value > best_value:
            best_value = value
            best_name = exp_dir.name
            best_overrides = payload.get("overrides", {})

    if not best_name:
        raise ValueError(f"No experiments with metrics found in {root}")

    return best_name, best_value, best_overrides


def promote_best(
    base_config_path: str | Path,
    sweep_out: str | Path,
    *,
    metric: str = "sharpe",
    dest: str | Path | None = None,
) -> PromoteResult:
    """Write a promoted config YAML applying the best sweep overrides."""
    base_path = Path(base_config_path)
    base_raw = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
    name, value, overrides = find_best_experiment(sweep_out, metric=metric)
    merged = apply_overrides(base_raw, overrides)

    promoted_path: Path | None = None
    if dest is not None:
        promoted_path = Path(dest)
        promoted_path.parent.mkdir(parents=True, exist_ok=True)
        promoted_path.write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")
        logger.info(
            "Promoted experiment %s (%s=%.4f) -> %s",
            name,
            metric,
            value,
            promoted_path,
        )

    # Validate merged config parses cleanly.
    AppConfig.model_validate(merged)

    return PromoteResult(
        experiment=name,
        metric=metric,
        metric_value=value,
        overrides=overrides,
        promoted_config_path=promoted_path,
    )

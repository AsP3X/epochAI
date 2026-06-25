"""Live session health checks."""

from __future__ import annotations

from dataclasses import dataclass

from epoch_ai.config.settings import AppConfig
from epoch_ai.execution.kill_switch import KillSwitch
from epoch_ai.models.registry import ModelRegistry


@dataclass(slots=True)
class LiveHealth:
    ready: bool
    model_loaded: bool
    models_available: int
    kill_switch_halted: bool
    min_buffer_bars: int
    issues: list[str]


def check_live_health(
    config: AppConfig,
    *,
    buffer_bars: int = 0,
    model_version: str | None = None,
) -> LiveHealth:
    """Return readiness for live predict/trade."""
    issues: list[str] = []
    registry = ModelRegistry(config.model.model_dir)
    versions = registry.list_versions()
    if not versions:
        issues.append("No trained models in registry; run train first.")

    if model_version:
        labels = {v["label"] for v in versions}
        if model_version not in labels:
            issues.append(f"Model {model_version} not found.")

    min_bars = max(config.execution.min_buffer_bars, config.walk_forward.initial_train_period)
    if buffer_bars and buffer_bars < min_bars:
        issues.append(f"Buffer {buffer_bars} < required {min_bars}.")

    ks = KillSwitch(config.execution.kill_switch_path)
    halted = ks.is_halted()
    if halted:
        issues.append(f"Kill switch active: {ks.read().reason}")

    return LiveHealth(
        ready=len(issues) == 0,
        model_loaded=bool(versions),
        models_available=len(versions),
        kill_switch_halted=halted,
        min_buffer_bars=min_bars,
        issues=issues,
    )

"""Coarse scheduled retrain helpers and holdout resolution."""

from __future__ import annotations

from epoch_ai.config.settings import AppConfig


def with_coarse_walk_forward(config: AppConfig) -> AppConfig:
    """Return a copy with larger walk-forward steps for scheduled auto-retrain."""
    if not config.adaptation.enabled:
        return config
    wf = config.walk_forward.model_copy(
        update={
            "step_size": config.adaptation.coarse_step_size,
            "retrain_frequency": config.adaptation.coarse_retrain_frequency,
        }
    )
    return config.model_copy(update={"walk_forward": wf})


def resolved_holdout_bars(config: AppConfig) -> int:
    """Final holdout slice size (never trained on)."""
    return config.adaptation.resolved_holdout_bars(config.promotion)


def trim_training_rows(config: AppConfig, n_rows: int) -> int:
    """Row count after excluding the final holdout tail."""
    holdout = resolved_holdout_bars(config)
    if holdout <= 0 or n_rows <= holdout:
        return n_rows
    return n_rows - holdout

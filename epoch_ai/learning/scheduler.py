"""Simple periodic retrain scheduler."""

from __future__ import annotations

import time

from epoch_ai.config.settings import AppConfig
from epoch_ai.learning.retrain_job import RetrainResult, run_retrain
from epoch_ai.utils.logging import get_logger

logger = get_logger(__name__)


def run_retrain_scheduler(
    config: AppConfig,
    *,
    interval_hours: float = 24.0,
    min_new_samples: int = 50,
    max_cycles: int | None = None,
) -> list[RetrainResult]:
    """Run retrain on a fixed interval until interrupted or ``max_cycles`` reached."""
    results: list[RetrainResult] = []
    cycle = 0
    while max_cycles is None or cycle < max_cycles:
        cycle += 1
        logger.info("Scheduler cycle %d starting retrain.", cycle)
        result = run_retrain(config, min_new_samples=min_new_samples)
        results.append(result)
        if max_cycles is not None and cycle >= max_cycles:
            break
        logger.info("Sleeping %.1f hours until next retrain.", interval_hours)
        time.sleep(interval_hours * 3600)
    return results

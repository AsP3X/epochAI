"""Simple periodic retrain scheduler."""

from __future__ import annotations

import time

from epoch_ai.config.settings import AppConfig
from epoch_ai.learning.promotion import AutoPromoteResult, auto_retrain_and_promote
from epoch_ai.learning.retrain_job import RetrainResult, run_retrain
from epoch_ai.utils.logging import get_logger

logger = get_logger(__name__)


def run_retrain_scheduler(
    config: AppConfig,
    *,
    interval_hours: float = 24.0,
    min_new_samples: int = 50,
    max_cycles: int | None = None,
    promote: bool = False,
) -> list[RetrainResult | AutoPromoteResult]:
    """Run retrain on a fixed interval until interrupted or ``max_cycles`` reached.

    Args:
        config: Application configuration.
        interval_hours: Sleep between cycles.
        min_new_samples: Minimum joined SQLite rows for the plain log/parquet retrain.
        max_cycles: Stop after N cycles (``None`` = run until interrupted).
        promote: When ``True`` each cycle runs the **challenger/champion** pipeline
            (:func:`auto_retrain_and_promote`) so the live model only changes when it
            improves out-of-sample; otherwise it runs the plain retrain job.
    """
    results: list[RetrainResult | AutoPromoteResult] = []
    cycle = 0
    while max_cycles is None or cycle < max_cycles:
        cycle += 1
        if promote:
            logger.info("Scheduler cycle %d starting auto-retrain (promote-if-better).", cycle)
            results.append(auto_retrain_and_promote(config))
        else:
            logger.info("Scheduler cycle %d starting retrain.", cycle)
            results.append(run_retrain(config, min_new_samples=min_new_samples))
        if max_cycles is not None and cycle >= max_cycles:
            break
        logger.info("Sleeping %.1f hours until next cycle.", interval_hours)
        time.sleep(interval_hours * 3600)
    return results

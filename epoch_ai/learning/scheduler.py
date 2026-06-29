"""Simple periodic retrain scheduler."""

from __future__ import annotations

import time

from epoch_ai.config.settings import AppConfig
from epoch_ai.learning.adaptation import with_coarse_walk_forward
from epoch_ai.learning.policy_promotion import PolicyPromoteResult, auto_train_and_promote_policy
from epoch_ai.learning.promotion import AutoPromoteResult, auto_retrain_and_promote
from epoch_ai.learning.retrain_job import RetrainResult, run_retrain
from epoch_ai.utils.logging import get_logger

logger = get_logger(__name__)


def run_retrain_scheduler(
    config: AppConfig,
    *,
    interval_hours: float | None = None,
    min_new_samples: int = 50,
    max_cycles: int | None = None,
    promote: bool = False,
    promote_policy: bool = False,
) -> list[RetrainResult | AutoPromoteResult | PolicyPromoteResult]:
    """Run retrain on a fixed interval until interrupted or ``max_cycles`` reached.

    Args:
        config: Application configuration.
        interval_hours: Sleep between cycles (default: ``adaptation.schedule_interval_hours``).
        min_new_samples: Minimum joined SQLite rows for the plain log/parquet retrain.
        max_cycles: Stop after N cycles (``None`` = run until interrupted).
        promote: When ``True`` each cycle runs the **challenger/champion** pipeline
            (:func:`auto_retrain_and_promote`) so the live model only changes when it
            improves out-of-sample; otherwise it runs the plain retrain job.
        promote_policy: When ``True`` (and ``promote``), also train/promote the PPO policy.
    """
    sleep_hours = (
        interval_hours
        if interval_hours is not None
        else config.adaptation.schedule_interval_hours
    )
    results: list[RetrainResult | AutoPromoteResult | PolicyPromoteResult] = []
    cycle = 0
    while max_cycles is None or cycle < max_cycles:
        cycle += 1
        cycle_config = with_coarse_walk_forward(config) if promote else config
        if promote:
            logger.info("Scheduler cycle %d starting auto-retrain (promote-if-better).", cycle)
            results.append(auto_retrain_and_promote(cycle_config))
            if promote_policy and config.rl.enabled:
                logger.info("Scheduler cycle %d starting policy auto-train.", cycle)
                results.append(auto_train_and_promote_policy(cycle_config))
        else:
            logger.info("Scheduler cycle %d starting retrain.", cycle)
            results.append(run_retrain(cycle_config, min_new_samples=min_new_samples))
        if max_cycles is not None and cycle >= max_cycles:
            break
        logger.info("Sleeping %.1f hours until next cycle.", sleep_hours)
        time.sleep(sleep_hours * 3600)
    return results

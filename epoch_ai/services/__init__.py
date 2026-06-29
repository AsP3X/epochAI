"""Public service layer for training and running epochAI.

Interfaces (CLI today; Telegram / website later) should depend on these services
instead of calling engines directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from epoch_ai.services.types import PredictionResult, RuntimeStatus

if TYPE_CHECKING:
    from epoch_ai.services.runtime import RuntimeService
    from epoch_ai.services.training import TrainingService, TrainResult

__all__ = [
    "PredictionResult",
    "RuntimeService",
    "RuntimeStatus",
    "TrainResult",
    "TrainingService",
]


def __getattr__(name: str):
    if name == "RuntimeService":
        from epoch_ai.services.runtime import RuntimeService

        return RuntimeService
    if name == "TrainingService":
        from epoch_ai.services.training import TrainingService

        return TrainingService
    if name == "TrainResult":
        from epoch_ai.services.training import TrainResult

        return TrainResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

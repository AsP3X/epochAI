"""Public service layer for training and running epochAI.

Interfaces (CLI today; Telegram / website later) should depend on these services
instead of calling engines directly.
"""

from __future__ import annotations

from epoch_ai.services.runtime import RuntimeService
from epoch_ai.services.training import TrainingService, TrainResult
from epoch_ai.services.types import PredictionResult, RuntimeStatus

__all__ = [
    "PredictionResult",
    "RuntimeService",
    "RuntimeStatus",
    "TrainResult",
    "TrainingService",
]

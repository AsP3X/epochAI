"""Sample-weighting helpers for training (shared across engine + retrain job).

Centralising the recency-decay logic keeps the progressive walk-forward engine and
the periodic retrain job consistent: both emphasise recent regimes the same way
instead of one path silently training unweighted.
"""

from __future__ import annotations

import numpy as np


def recency_weights(n: int, half_life: int | None) -> np.ndarray | None:
    """Exponentially-decayed per-row weights, newest row weighted ``1.0``.

    Args:
        n: Number of training rows, assumed in **chronological order** (oldest first).
        half_life: Decay half-life in rows. ``None``/``0`` disables weighting.

    Returns:
        A float64 array of length ``n`` (weight 1.0 for the most recent row, decaying
        into the past), or ``None`` when weighting is disabled.
    """
    # Human: Disabled (or empty) -> let LightGBM treat every row equally.
    # Agent: RETURNS None when half_life falsy; CAUSAL weights depend only on row age.
    if not half_life or n <= 0:
        return None
    age = np.arange(n)[::-1]  # 0 = most recent row
    return np.power(0.5, age / float(half_life)).astype(np.float64)

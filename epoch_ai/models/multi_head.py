"""Multi-horizon output layout, targets, and losses for ``evolved_nn``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from epoch_ai.config.settings import PredictionConfig


@dataclass(frozen=True, slots=True)
class MultiHeadSpec:
    """Flat output layout: per horizon, ``len(quantiles)`` return quantiles + 1 direction logit."""

    horizons: tuple[int, ...]
    quantiles: tuple[float, ...]

    @classmethod
    def from_prediction(cls, prediction: PredictionConfig) -> MultiHeadSpec:
        return cls(tuple(prediction.horizons), tuple(prediction.quantiles))

    @property
    def n_outputs(self) -> int:
        return len(self.horizons) * (len(self.quantiles) + 1)

    @property
    def primary_horizon(self) -> int:
        return self.horizons[-1]

    def head_offset(self, horizon: int) -> int:
        idx = self.horizons.index(horizon)
        return idx * (len(self.quantiles) + 1)

    def direction_index(self, horizon: int) -> int:
        return self.head_offset(horizon) + len(self.quantiles)

    def quantile_slice(self, horizon: int) -> slice:
        start = self.head_offset(horizon)
        return slice(start, start + len(self.quantiles))

    def to_dict(self) -> dict[str, object]:
        return {"horizons": list(self.horizons), "quantiles": list(self.quantiles)}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> MultiHeadSpec:
        return cls(
            tuple(int(h) for h in payload["horizons"]),  # type: ignore[arg-type]
            tuple(float(q) for q in payload["quantiles"]),  # type: ignore[arg-type]
        )


def targets_to_matrix(targets: pd.DataFrame, spec: MultiHeadSpec) -> np.ndarray:
    """Build ``(n_rows, n_outputs)`` training matrix from ``build_multi_horizon_targets`` output."""
    n = len(targets)
    q = len(spec.quantiles)
    out = np.empty((n, spec.n_outputs), dtype=np.float32)
    for h in spec.horizons:
        base = spec.head_offset(h)
        ret = targets[f"ret_{h}"].to_numpy(dtype=np.float32)
        tgt = targets[f"target_{h}"].to_numpy(dtype=np.float32)
        for k in range(q):
            out[:, base + k] = ret
        out[:, base + q] = tgt
    return out


def sort_quantile_predictions(preds: np.ndarray, spec: MultiHeadSpec) -> np.ndarray:
    """Enforce monotone quantiles per horizon (p10 <= p50 <= p90)."""
    out = preds.copy()
    for h in spec.horizons:
        sl = spec.quantile_slice(h)
        out[:, sl] = np.sort(out[:, sl], axis=1)
    return out


def multi_head_train_loss(
    logits,
    y_true,
    spec: MultiHeadSpec,
    *,
    pos_weights: list | None = None,
) -> object:
    """Mean loss across horizons: pinball on returns + BCE on direction logits."""
    import torch
    import torch.nn as nn

    total = torch.zeros((), device=logits.device, dtype=logits.dtype)
    q = len(spec.quantiles)
    for i, h in enumerate(spec.horizons):
        base = spec.head_offset(h)
        q_preds = logits[:, base : base + q]
        q_sorted, _ = torch.sort(q_preds, dim=1)
        q_true = y_true[:, base : base + q]
        for k, qt in enumerate(spec.quantiles):
            err = q_true[:, k] - q_sorted[:, k]
            loss_q = torch.maximum(qt * err, (qt - 1.0) * err)
            total = total + loss_q.mean()
        dir_logit = logits[:, base + q]
        dir_true = y_true[:, base + q]
        pw = None if pos_weights is None else pos_weights[i]
        bce = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pw)
        total = total + bce(dir_logit, dir_true)
    return total / max(1, len(spec.horizons))


def multi_head_val_loss(
    logits_np: np.ndarray,
    y_np: np.ndarray,
    spec: MultiHeadSpec,
    *,
    primary_horizon: int,
) -> float:
    """Validation score for early stopping (primary direction logloss + mean pinball)."""
    from sklearn.metrics import log_loss

    logits_np = sort_quantile_predictions(logits_np, spec)
    pinballs = []
    for h in spec.horizons:
        base = spec.head_offset(h)
        for k, qt in enumerate(spec.quantiles):
            err = y_np[:, base + k] - logits_np[:, base + k]
            pinballs.append(float(np.maximum(qt * err, (qt - 1.0) * err).mean()))
    dir_idx = spec.direction_index(primary_horizon)
    probs = 1.0 / (1.0 + np.exp(-logits_np[:, dir_idx]))
    dir_loss = float(log_loss(y_np[:, dir_idx], probs, labels=[0, 1]))
    return dir_loss + float(np.mean(pinballs))


def parse_structured_predictions(
    logits: np.ndarray,
    spec: MultiHeadSpec,
    *,
    primary_horizon: int,
) -> dict[int, dict[str, float | np.ndarray]]:
    """Parse flat logits into per-horizon quantile returns and direction probabilities."""
    logits = sort_quantile_predictions(logits, spec)
    out: dict[int, dict[str, float | np.ndarray]] = {}
    q = len(spec.quantiles)
    for h in spec.horizons:
        base = spec.head_offset(h)
        dir_logit = logits[:, base + q] if logits.ndim == 2 else float(logits[base + q])
        if logits.ndim == 2:
            p_up = 1.0 / (1.0 + np.exp(-dir_logit))
            rets = {f"q{int(qt * 100)}": logits[:, base + k] for k, qt in enumerate(spec.quantiles)}
            rets["p_up"] = p_up
            rets["exp_return"] = rets.get("q50", logits[:, base + q // 2])
        else:
            p_up = float(1.0 / (1.0 + np.exp(-dir_logit)))
            rets = {
                f"q{int(qt * 100)}": float(logits[base + k]) for k, qt in enumerate(spec.quantiles)
            }
            rets["p_up"] = p_up
            rets["exp_return"] = rets.get("q50", float(logits[base + q // 2]))
        out[h] = rets
    return out

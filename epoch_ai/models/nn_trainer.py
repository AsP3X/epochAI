"""PyTorch MLP training for a fixed :class:`~epoch_ai.models.nn_genome.NNGenome`."""

from __future__ import annotations

import os
from dataclasses import dataclass

# Human: Limit BLAS/OpenMP threads before importing torch to avoid segfaults when
#        pytest has already loaded LightGBM/sklearn in the same process.
# Agent: SETS OMP/MKL/OPENBLAS=1; CAUSAL import-time side effect for evolved_nn only.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.preprocessing import StandardScaler

from epoch_ai.config.settings import ModelConfig
from epoch_ai.models.nn_genome import NNGenome
from epoch_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class TrainResult:
    """Outcome of training one genome."""

    val_loss: float
    best_epoch: int
    state_dict: dict[str, object]
    scaler: StandardScaler


def _import_torch():
    try:
        import torch
        import torch.nn as nn

        torch.set_num_threads(1)
    except ImportError as exc:  # pragma: no cover - exercised when torch missing
        raise ImportError(
            "model.backend='evolved_nn' requires PyTorch. "
            "Install with `pip install torch` or `pip install -r requirements-optional.txt`."
        ) from exc
    return torch, nn


def resolve_device(config: ModelConfig):
    """Return a ``torch.device``, falling back to CPU when CUDA is unavailable."""
    torch, _ = _import_torch()
    requested = config.device
    if requested in ("cuda", "gpu"):
        if torch.cuda.is_available():
            device_id = config.gpu_device_id
            if device_id >= 0:
                return torch.device(f"cuda:{device_id}")
            return torch.device("cuda")
        logger.warning("CUDA requested but unavailable; using CPU for evolved_nn.")
    return torch.device("cpu")


def build_mlp(input_dim: int, genome: NNGenome, *, task: str):
    """Construct a feed-forward network from ``genome``."""
    _, nn = _import_torch()
    layers: list = []
    prev = input_dim
    for hidden in genome.hidden_sizes:
        layers.append(nn.Linear(prev, hidden))
        if genome.use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden))
        layers.append(nn.ReLU())
        if genome.dropout > 0.0:
            layers.append(nn.Dropout(genome.dropout))
        prev = hidden
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def _scale_pos_weight(y: np.ndarray) -> float:
    labels = y.ravel()
    n_pos = float((labels > 0.5).sum())
    n_neg = float((labels <= 0.5).sum())
    if n_pos <= 0.0 or n_neg <= 0.0:
        return 1.0
    return n_neg / n_pos


def train_genome(
    x: np.ndarray,
    y: np.ndarray,
    genome: NNGenome,
    config: ModelConfig,
    *,
    task: str,
    sample_weight: np.ndarray | None,
    val_fraction: float,
    split: int | None = None,
    refit_full: bool = False,
) -> TrainResult:
    """Fit weights for ``genome`` with Adam and time-ordered early stopping.

    Args:
        x: Feature matrix in chronological order.
        y: Target aligned to ``x``.
        genome: Architecture hyper-parameters.
        config: Model configuration (device, nn/evolution nested settings).
        task: ``classification`` or ``regression``.
        sample_weight: Optional per-row weights.
        val_fraction: Fraction of most-recent rows held out for validation.
        split: Explicit train/val split index; computed from ``val_fraction`` when ``None``.
        refit_full: When ``True``, retrain on all rows for ``best_epoch`` rounds.
    """
    torch, nn = _import_torch()
    nn_cfg = config.nn
    device = resolve_device(config)

    if split is None:
        has_val = 0.0 < val_fraction < 0.5 and len(x) >= 200
        split = int(len(x) * (1.0 - val_fraction)) if has_val else len(x)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[:split])
    y_train = y[:split].astype(np.float32)
    w_train = None if sample_weight is None else sample_weight[:split].astype(np.float32)

    has_val = split < len(x)
    if has_val:
        x_val = scaler.transform(x[split:])
        y_val = y[split:].astype(np.float32)
    else:
        x_val = None
        y_val = None

    model = build_mlp(x_train.shape[1], genome, task=task).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=genome.learning_rate,
        weight_decay=genome.weight_decay,
    )

    is_classification = task == "classification"
    pos_weight = None
    if is_classification and config.class_weight == "balanced":
        pos_weight = torch.tensor([_scale_pos_weight(y_train)], device=device)

    if is_classification:
        criterion: nn.Module = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    else:
        criterion = nn.MSELoss(reduction="none")

    x_train_t = torch.from_numpy(x_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device).view(-1, 1)
    if w_train is not None:
        w_train_t = torch.from_numpy(w_train).float().to(device)
    else:
        w_train_t = None

    if has_val and x_val is not None and y_val is not None:
        x_val_t = torch.from_numpy(x_val).float().to(device)
        y_val_t = torch.from_numpy(y_val).float().to(device).view(-1, 1)
    else:
        x_val_t = None
        y_val_t = None

    best_state: dict[str, object] | None = None
    best_val = float("inf")
    best_epoch = 0
    patience_left = nn_cfg.patience

    for epoch in range(nn_cfg.max_epochs):
        model.train()
        n = len(x_train_t)
        indices = torch.randperm(n, device=device)
        for start in range(0, n, nn_cfg.batch_size):
            idx = indices[start : start + nn_cfg.batch_size]
            batch_x = x_train_t[idx]
            batch_y = y_train_t[idx]
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss_vec = criterion(logits, batch_y)
            if w_train_t is not None:
                batch_w = w_train_t[idx]
                loss = (loss_vec.view(-1) * batch_w).mean()
            else:
                loss = loss_vec.mean()
            loss.backward()
            optimizer.step()

        if x_val_t is None or y_val_t is None:
            if epoch + 1 >= min(nn_cfg.max_epochs // 4, 30):
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch + 1
                best_val = 0.0
                break
            continue

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val_t)
            if is_classification:
                val_probs = torch.sigmoid(val_logits).cpu().numpy().ravel()
                val_loss = log_loss(y_val, val_probs, labels=[0, 1])
            else:
                val_preds = val_logits.cpu().numpy().ravel()
                val_loss = mean_squared_error(y_val, val_preds)

        if val_loss + 1e-6 < best_val:
            best_val = float(val_loss)
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = nn_cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = max(1, nn_cfg.max_epochs // 4)

    if refit_full and split < len(x):
        full_scaler = StandardScaler()
        x_full = full_scaler.fit_transform(x)
        y_full = y.astype(np.float32)
        w_full = None if sample_weight is None else sample_weight.astype(np.float32)
        model = build_mlp(x_full.shape[1], genome, task=task).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=genome.learning_rate,
            weight_decay=genome.weight_decay,
        )
        x_full_t = torch.from_numpy(x_full).float().to(device)
        y_full_t = torch.from_numpy(y_full).float().to(device).view(-1, 1)
        w_full_t = None if w_full is None else torch.from_numpy(w_full).float().to(device)
        for _ in range(best_epoch):
            model.train()
            n = len(x_full_t)
            indices = torch.randperm(n, device=device)
            for start in range(0, n, nn_cfg.batch_size):
                idx = indices[start : start + nn_cfg.batch_size]
                optimizer.zero_grad(set_to_none=True)
                logits = model(x_full_t[idx])
                loss_vec = criterion(logits, y_full_t[idx])
                if w_full_t is not None:
                    loss = (loss_vec.view(-1) * w_full_t[idx]).mean()
                else:
                    loss = loss_vec.mean()
                loss.backward()
                optimizer.step()
        scaler = full_scaler
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if x_val_t is not None and y_val is not None and best_state is not None:
        model.load_state_dict(best_state)  # type: ignore[arg-type]
        model.eval()
        with torch.no_grad():
            val_logits = model(x_val_t)
            if is_classification:
                val_probs = torch.sigmoid(val_logits).cpu().numpy().ravel()
                best_val = float(log_loss(y_val, val_probs, labels=[0, 1]))
            else:
                val_preds = val_logits.cpu().numpy().ravel()
                best_val = float(mean_squared_error(y_val, val_preds))
    elif best_val == float("inf"):
        best_val = 0.0

    return TrainResult(
        val_loss=best_val,
        best_epoch=best_epoch,
        state_dict=best_state,
        scaler=scaler,
    )


def predict_genome(
    x: np.ndarray,
    genome: NNGenome,
    state_dict: dict[str, object],
    scaler: StandardScaler,
    config: ModelConfig,
    *,
    task: str,
) -> np.ndarray:
    """Run a trained genome forward and return probabilities or regression outputs."""
    torch, _ = _import_torch()
    device = resolve_device(config)
    x_scaled = scaler.transform(x)
    model = build_mlp(x_scaled.shape[1], genome, task=task).to(device)
    model.load_state_dict(state_dict)  # type: ignore[arg-type]
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x_scaled).float().to(device))
        if task == "classification":
            return torch.sigmoid(logits).cpu().numpy().ravel()
        return logits.cpu().numpy().ravel()


def permutation_importance(
    x: np.ndarray,
    y: np.ndarray,
    genome: NNGenome,
    state_dict: dict[str, object],
    scaler: StandardScaler,
    config: ModelConfig,
    *,
    task: str,
    feature_names: list[str],
    max_features: int = 40,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Estimate feature importance by shuffling each column on a validation slice."""
    if task != "classification":
        return {name: 0.0 for name in feature_names}

    rng = rng or np.random.default_rng(0)
    baseline_probs = predict_genome(x, genome, state_dict, scaler, config, task=task)
    baseline = float(log_loss(y, baseline_probs, labels=[0, 1]))

    names = feature_names
    if len(names) > max_features:
        idx = rng.choice(len(names), size=max_features, replace=False)
        names = [feature_names[i] for i in sorted(idx)]

    importances: dict[str, float] = {}
    for name in names:
        col_idx = feature_names.index(name)
        x_perm = x.copy()
        rng.shuffle(x_perm[:, col_idx])
        perm_probs = predict_genome(x_perm, genome, state_dict, scaler, config, task=task)
        perm_loss = float(log_loss(y, perm_probs, labels=[0, 1]))
        importances[name] = max(0.0, perm_loss - baseline)

    for name in feature_names:
        importances.setdefault(name, 0.0)
    return importances

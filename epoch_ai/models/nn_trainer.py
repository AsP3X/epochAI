"""PyTorch MLP training for a fixed :class:`~epoch_ai.models.nn_genome.NNGenome`."""

from __future__ import annotations

import os
import sysconfig
import threading
from contextlib import nullcontext
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

from epoch_ai.config.settings import EvolutionConfig, ModelConfig
from epoch_ai.models.multi_head import (
    MultiHeadSpec,
    multi_head_train_loss,
    multi_head_val_loss_torch,
)
from epoch_ai.models.nn_genome import NNGenome
from epoch_ai.utils.logging import get_logger

logger = get_logger(__name__)

_COMPILE_SKIP_LOGGED = False
_CUDA_RUNTIME_CONFIGURED = False
_CUDA_TRAINING_LOGGED = False

_thread_local = threading.local()


@dataclass(slots=True)
class TrainResult:
    """Outcome of training one genome."""

    val_loss: float
    best_epoch: int
    state_dict: dict[str, object]
    scaler: StandardScaler


@dataclass(slots=True)
class TrainingDataCache:
    """Device-resident tensors shared across genomes within one ``fit()`` call."""

    device: object
    split: int
    has_val: bool
    scaler: StandardScaler
    is_classification: bool
    pos_weight: object | None
    x_train_t: object
    y_train_t: object
    w_train_t: object | None
    x_val_t: object | None
    y_val_t: object | None
    y_val_np: np.ndarray | None


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
    """Return a ``torch.device``; ``auto`` prefers CUDA when available."""
    torch, _ = _import_torch()
    requested = config.device
    if requested == "auto":
        if torch.cuda.is_available():
            device_id = config.gpu_device_id
            if device_id >= 0:
                return torch.device(f"cuda:{device_id}")
            return torch.device("cuda")
        return torch.device("cpu")
    if requested in ("cuda", "gpu"):
        if torch.cuda.is_available():
            device_id = config.gpu_device_id
            if device_id >= 0:
                return torch.device(f"cuda:{device_id}")
            return torch.device("cuda")
        logger.warning("CUDA requested but unavailable; using CPU for evolved_nn.")
    return torch.device("cpu")


def resolve_cuda_worker_cap(vram_gb: float, evolution: EvolutionConfig) -> int:
    """Map GPU VRAM (GB) to parallel evolution workers using config tier tables."""
    if not evolution.cuda_auto_workers:
        return min(evolution.cuda_worker_cap_fallback, evolution.cuda_worker_cap_max)
    caps = evolution.cuda_worker_caps
    tiers = evolution.cuda_worker_vram_gb
    cap = caps[0]
    for i, tier in enumerate(tiers):
        if vram_gb >= tier:
            cap = caps[i + 1]
    return min(cap, evolution.cuda_worker_cap_max)


def configure_cuda_runtime(device, config: ModelConfig) -> None:
    """Apply config-driven CUDA matmul/cudnn settings once per process."""
    global _CUDA_RUNTIME_CONFIGURED
    if _CUDA_RUNTIME_CONFIGURED or getattr(device, "type", "") != "cuda":
        return
    torch, _ = _import_torch()
    cuda_cfg = config.cuda
    if cuda_cfg.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if cuda_cfg.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    # Human: route fp32 matmuls through TF32/bf16 tensor cores; biggest single throughput win
    #        for the small dense MLPs in evolution. "highest" keeps strict fp32.
    # Agent: CONFIG cuda.matmul_precision; safe no-op on non-Ampere GPUs.
    if cuda_cfg.matmul_precision and cuda_cfg.matmul_precision != "highest":
        try:
            torch.set_float32_matmul_precision(cuda_cfg.matmul_precision)
        except (ValueError, AttributeError):
            logger.warning(
                "Unsupported matmul_precision=%r; leaving Torch default.",
                cuda_cfg.matmul_precision,
            )
    _CUDA_RUNTIME_CONFIGURED = True
    props = torch.cuda.get_device_properties(device)
    vram_gb = props.total_memory / (1024**3)
    logger.info(
        "CUDA runtime: tf32=%s cudnn.benchmark=%s matmul=%s (%s, %.1f GB VRAM).",
        cuda_cfg.allow_tf32,
        cuda_cfg.cudnn_benchmark,
        cuda_cfg.matmul_precision,
        props.name,
        vram_gb,
    )


def effective_training_batch_size(nn_cfg, n_train: int, device) -> int:
    """Resolve per-epoch minibatch size; scale up on CUDA when auto-batch is enabled."""
    base = nn_cfg.batch_size
    if getattr(device, "type", "") != "cuda" or n_train <= 0:
        return base
    if not nn_cfg.cuda_auto_batch:
        return min(base, n_train)
    # Human: tiny batches leave the GPU idle between kernel launches; target configurable steps.
    # Agent: READS nn.cuda_batches_per_epoch + cuda_batch_cap; CAUSAL same epoch schedule.
    target_steps = nn_cfg.cuda_batches_per_epoch
    scaled = max(base, (n_train + target_steps - 1) // target_steps)
    scaled = min(scaled, nn_cfg.cuda_batch_cap, n_train)
    return max(base, scaled)


def evolution_max_workers(config: ModelConfig, population_size: int) -> int:
    """Resolve parallel candidate worker count for one evolution generation."""
    evolution = config.evolution
    if evolution.max_workers is not None:
        return min(evolution.max_workers, population_size)
    device = resolve_device(config)
    if device.type == "cuda":
        if evolution.cuda_auto_workers:
            torch, _ = _import_torch()
            props = torch.cuda.get_device_properties(device)
            cap = resolve_cuda_worker_cap(props.total_memory / (1024**3), evolution)
        else:
            cap = min(evolution.cuda_worker_cap_fallback, evolution.cuda_worker_cap_max)
        return min(cap, population_size)
    return min(os.cpu_count() or 1, population_size)


def _cuda_stream(device):
    """Per-thread CUDA stream so parallel candidates can share one GPU safely."""
    torch, _ = _import_torch()
    if device.type != "cuda":
        return nullcontext()
    stream = getattr(_thread_local, "cuda_stream", None)
    if stream is None or stream.device != device:
        stream = torch.cuda.Stream(device=device)
        _thread_local.cuda_stream = stream
    return torch.cuda.stream(stream)


def _min_train_batch(genome: NNGenome) -> int:
    """BatchNorm1d requires batch size > 1 during training."""
    return 2 if genome.use_batch_norm else 1


def build_mlp(input_dim: int, genome: NNGenome, *, task: str, n_outputs: int = 1):
    """Construct a feed-forward network from ``genome``.

    Args:
        n_outputs: Output neuron count (1 for legacy single-head; multi-horizon uses
            ``prediction.n_outputs``).
    """
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
    layers.append(nn.Linear(prev, n_outputs))
    return nn.Sequential(*layers)


def _triton_available() -> bool:
    """Return whether Triton is importable (required by torch.compile on CUDA)."""
    try:
        import triton  # noqa: F401, PLC0415

        return True
    except ImportError:
        return False


def _python_dev_headers_available() -> bool:
    """Return whether Python.h is present (required by Triton JIT on Linux)."""
    include = sysconfig.get_path("include")
    if not include:
        return False
    return os.path.isfile(os.path.join(include, "Python.h"))


def _triton_compile_ready() -> bool:
    """Triton plus Python dev headers are required for torch.compile on CUDA."""
    return _triton_available() and _python_dev_headers_available()


def _maybe_compile(model, config: ModelConfig, device, *, warmup_batch=None):
    """Apply ``torch.compile`` on CUDA when configured and supported.

    Restricted to CUDA on the **main thread** only: parallel evolution trains candidates
    in a thread pool, and torch.compile/dynamo is not safe across worker threads (and
    fails on Windows with FX tracing errors even when Triton is installed).
    """
    if not config.nn.torch_compile or getattr(device, "type", "") != "cuda":
        return model
    if threading.current_thread() is not threading.main_thread():
        return model
    global _COMPILE_SKIP_LOGGED
    if not _triton_available():
        if not _COMPILE_SKIP_LOGGED:
            logger.info(
                "torch.compile disabled for evolved_nn: Triton is not installed "
                "(common on Windows CUDA builds)."
            )
            _COMPILE_SKIP_LOGGED = True
        return model
    if not _python_dev_headers_available():
        if not _COMPILE_SKIP_LOGGED:
            logger.info(
                "torch.compile disabled for evolved_nn: Python.h not found (install "
                "python3-dev / python3.12-dev on Linux, or set model.nn.torch_compile=false)."
            )
            _COMPILE_SKIP_LOGGED = True
        return model
    torch, _ = _import_torch()
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        return model
    try:
        compiled = compile_fn(model)
        # Human: compile() is lazy; one tiny forward pass surfaces Triton JIT errors early.
        # Agent: CALLS warmup_batch slice; RETURNS uncompiled model on dynamo/triton failure.
        if warmup_batch is not None and len(warmup_batch) > 0:
            use_amp = bool(config.nn.mixed_precision)
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    compiled(warmup_batch[: min(2, len(warmup_batch))])
        return compiled
    except Exception as exc:  # pragma: no cover - backend-specific compile failures
        if not _COMPILE_SKIP_LOGGED:
            logger.info(
                "torch.compile disabled for evolved_nn after compile failure: %s",
                exc,
            )
            _COMPILE_SKIP_LOGGED = True
        return model


def _scale_pos_weight(y: np.ndarray) -> float:
    labels = y.ravel()
    n_pos = float((labels > 0.5).sum())
    n_neg = float((labels <= 0.5).sum())
    if n_pos <= 0.0 or n_neg <= 0.0:
        return 1.0
    return n_neg / n_pos


def build_training_cache(
    x: np.ndarray,
    y: np.ndarray,
    config: ModelConfig,
    *,
    task: str,
    sample_weight: np.ndarray | None,
    val_fraction: float,
    split: int | None = None,
    multi_head: MultiHeadSpec | None = None,
) -> TrainingDataCache:
    """Fit the scaler once and upload train/val tensors for reuse across genomes."""
    torch, _ = _import_torch()
    device = resolve_device(config)
    configure_cuda_runtime(device, config)

    if split is None:
        has_val = 0.0 < val_fraction < 0.5 and len(x) >= 200
        split = int(len(x) * (1.0 - val_fraction)) if has_val else len(x)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[:split])
    y_train = y[:split].astype(np.float32)
    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)
    w_train = None if sample_weight is None else sample_weight[:split].astype(np.float32)

    has_val = split < len(x)
    y_val_np = None
    if has_val:
        x_val = scaler.transform(x[split:])
        y_val_np = y[split:].astype(np.float32)
        if y_val_np.ndim == 1:
            y_val_np = y_val_np.reshape(-1, 1)
    else:
        x_val = None

    is_classification = task == "classification"
    pos_weight = None
    if is_classification and config.class_weight == "balanced":
        if multi_head is not None:
            pos_weight = [
                torch.tensor(
                    [_scale_pos_weight(y_train[:, multi_head.direction_index(h)])],
                    device=device,
                )
                for h in multi_head.horizons
            ]
        else:
            pos_weight = torch.tensor([_scale_pos_weight(y_train)], device=device)

    x_train_t = torch.from_numpy(x_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device)
    w_train_t = None if w_train is None else torch.from_numpy(w_train).float().to(device)

    if has_val and x_val is not None and y_val_np is not None:
        x_val_t = torch.from_numpy(x_val).float().to(device)
        y_val_t = torch.from_numpy(y_val_np).float().to(device)
    else:
        x_val_t = None
        y_val_t = None

    return TrainingDataCache(
        device=device,
        split=split,
        has_val=has_val,
        scaler=scaler,
        is_classification=is_classification,
        pos_weight=pos_weight,
        x_train_t=x_train_t,
        y_train_t=y_train_t,
        w_train_t=w_train_t,
        x_val_t=x_val_t,
        y_val_t=y_val_t,
        y_val_np=y_val_np,
    )


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
    cache: TrainingDataCache | None = None,
    initial_state: dict[str, object] | None = None,
    multi_head: MultiHeadSpec | None = None,
    primary_horizon: int | None = None,
    max_epochs_override: int | None = None,
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
        cache: Optional pre-built device tensors (same split for every genome in one fit).
        initial_state: Optional warm-start weights when architecture matches ``genome``.
        max_epochs_override: Cap the early-stopping epoch budget below ``nn.max_epochs``
            (used by successive-halving proxy rungs); ``None`` keeps the configured value.
    """
    torch, nn = _import_torch()
    nn_cfg = config.nn
    # Human: successive-halving cheap rungs pass a reduced budget; full rungs pass None.
    # Agent: CAUSAL only changes how many epochs we train, not which bars are in train.
    max_epochs = nn_cfg.max_epochs if max_epochs_override is None else max(1, int(max_epochs_override))

    if cache is None:
        cache = build_training_cache(
            x,
            y,
            config,
            task=task,
            sample_weight=sample_weight,
            val_fraction=val_fraction,
            split=split,
            multi_head=multi_head,
        )

    device = cache.device
    configure_cuda_runtime(device, config)
    split = cache.split
    scaler = cache.scaler
    y_val = cache.y_val_np
    n_outputs = multi_head.n_outputs if multi_head is not None else 1
    if primary_horizon is None and multi_head is not None:
        primary_horizon = multi_head.horizons[-1]

    x_train_t = cache.x_train_t
    y_train_t = cache.y_train_t
    w_train_t = cache.w_train_t
    x_val_t = cache.x_val_t
    y_val_t = cache.y_val_t

    is_classification = cache.is_classification
    pos_weight = cache.pos_weight

    if is_classification:
        criterion: nn.Module | None = None if multi_head is not None else nn.BCEWithLogitsLoss(
            pos_weight=pos_weight if not isinstance(pos_weight, list) else None,
            reduction="none",
        )
    else:
        criterion = nn.MSELoss(reduction="none")

    use_amp = bool(nn_cfg.mixed_precision and getattr(device, "type", "") == "cuda")
    train_batch = effective_training_batch_size(nn_cfg, len(x_train_t), device)
    global _CUDA_TRAINING_LOGGED
    if getattr(device, "type", "") == "cuda" and not _CUDA_TRAINING_LOGGED:
        _CUDA_TRAINING_LOGGED = True
        logger.info(
            "CUDA training batch_size=%d (config=%d, auto_batch=%s, cap=%d).",
            train_batch,
            nn_cfg.batch_size,
            nn_cfg.cuda_auto_batch,
            nn_cfg.cuda_batch_cap,
        )

    with _cuda_stream(device):
        # Human: keep the uncompiled module for state I/O; torch.compile wraps it in
        #        _orig_mod and would otherwise prefix every state_dict key.
        # Agent: base_model owns params; compiled `model` shares them for forward/backward.
        base_model = build_mlp(
            int(x_train_t.shape[1]), genome, task=task, n_outputs=n_outputs
        ).to(device)
        if initial_state is not None:
            try:
                base_model.load_state_dict(initial_state)  # type: ignore[arg-type]
            except RuntimeError:
                pass
        model = _maybe_compile(base_model, config, device, warmup_batch=x_train_t)
        optimizer = torch.optim.Adam(
            base_model.parameters(),
            lr=genome.learning_rate,
            weight_decay=genome.weight_decay,
        )

        best_state: dict[str, object] | None = None
        best_val = float("inf")
        best_epoch = 0
        patience_left = nn_cfg.patience
        min_batch = _min_train_batch(genome)

        for epoch in range(max_epochs):
            model.train()
            n = len(x_train_t)
            indices = torch.randperm(n, device=device)
            for start in range(0, n, train_batch):
                idx = indices[start : start + train_batch]
                # Human: val tail can leave train_rows % batch_size == 1; BatchNorm rejects N=1.
                # Agent: SKIP batches smaller than min_batch; CAUSAL no effect at predict time.
                if len(idx) < min_batch:
                    continue
                batch_x = x_train_t[idx]
                batch_y = y_train_t[idx]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type="cuda",
                    enabled=use_amp,
                ):
                    logits = model(batch_x)
                    if multi_head is not None:
                        loss = multi_head_train_loss(
                            logits,
                            batch_y,
                            multi_head,
                            pos_weights=pos_weight if isinstance(pos_weight, list) else None,
                        )
                    else:
                        loss_vec = criterion(logits, batch_y)  # type: ignore[misc]
                        if w_train_t is not None:
                            loss = (loss_vec.view(-1) * w_train_t[idx]).mean()
                        else:
                            loss = loss_vec.mean()
                if multi_head is not None and w_train_t is not None:
                    loss = loss * w_train_t[idx].mean()
                loss.backward()
                optimizer.step()

            if x_val_t is None or y_val_t is None:
                if epoch + 1 >= min(max_epochs // 4, 30):
                    best_state = {
                        k: v.detach().clone() for k, v in base_model.state_dict().items()
                    }
                    best_epoch = epoch + 1
                    best_val = 0.0
                    break
                continue

            # Human: skip val on most epochs when interval > 1; still early-stop on scheduled checks.
            # Agent: CONFIG nn.val_check_interval; CAUSAL val ordering unchanged when interval=1.
            if (epoch + 1) % nn_cfg.val_check_interval != 0:
                continue

            model.eval()
            with torch.no_grad():
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    val_logits = model(x_val_t)
                if multi_head is not None and primary_horizon is not None and y_val_t is not None:
                    val_loss = multi_head_val_loss_torch(
                        val_logits,
                        y_val_t,
                        multi_head,
                        primary_horizon=primary_horizon,
                    )
                elif is_classification:
                    val_loss = float(
                        nn.functional.binary_cross_entropy_with_logits(
                            val_logits.float(),
                            y_val_t,
                            reduction="mean",
                        )
                    )
                else:
                    val_preds = val_logits.float().cpu().numpy().ravel()
                    val_loss = mean_squared_error(y_val, val_preds)

            if val_loss + 1e-6 < best_val:
                best_val = float(val_loss)
                best_epoch = epoch + 1
                best_state = {
                    k: v.detach().clone() for k, v in base_model.state_dict().items()
                }
                patience_left = nn_cfg.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is None:
            best_state = {k: v.detach().clone() for k, v in base_model.state_dict().items()}
            best_epoch = max(1, max_epochs // 4)

        if refit_full and split < len(x):
            full_scaler = StandardScaler()
            x_full = full_scaler.fit_transform(x)
            y_full = y.astype(np.float32)
            if y_full.ndim == 1:
                y_full = y_full.reshape(-1, 1)
            w_full = None if sample_weight is None else sample_weight.astype(np.float32)
            base_model = build_mlp(x_full.shape[1], genome, task=task, n_outputs=n_outputs).to(device)
            # Human: warm-start the full-window refit from the early-stopped best weights
            #        so Adam resumes near the optimum instead of from random init.
            # Agent: input_dim matches (same feature set); CAUSAL no leakage (weights only).
            if best_state is not None and x_full.shape[1] == int(x_train_t.shape[1]):
                try:
                    base_model.load_state_dict(best_state)  # type: ignore[arg-type]
                except RuntimeError:
                    pass
            x_full_t = torch.from_numpy(x_full).float().to(device)
            y_full_t = torch.from_numpy(y_full).float().to(device)
            w_full_t = None if w_full is None else torch.from_numpy(w_full).float().to(device)
            model = _maybe_compile(base_model, config, device, warmup_batch=x_full_t)
            optimizer = torch.optim.Adam(
                base_model.parameters(),
                lr=genome.learning_rate,
                weight_decay=genome.weight_decay,
            )
            full_batch = effective_training_batch_size(nn_cfg, len(x_full_t), device)
            for _ in range(best_epoch):
                model.train()
                n = len(x_full_t)
                indices = torch.randperm(n, device=device)
                for start in range(0, n, full_batch):
                    idx = indices[start : start + full_batch]
                    if len(idx) < min_batch:
                        continue
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", enabled=use_amp):
                        logits = model(x_full_t[idx])
                        if multi_head is not None:
                            loss = multi_head_train_loss(
                                logits,
                                y_full_t[idx],
                                multi_head,
                                pos_weights=pos_weight if isinstance(pos_weight, list) else None,
                            )
                        else:
                            loss_vec = criterion(logits, y_full_t[idx])  # type: ignore[misc]
                            if w_full_t is not None:
                                loss = (loss_vec.view(-1) * w_full_t[idx]).mean()
                            else:
                                loss = loss_vec.mean()
                    loss.backward()
                    optimizer.step()
            scaler = full_scaler
            best_state = {k: v.detach().clone() for k, v in base_model.state_dict().items()}

        if x_val_t is not None and y_val is not None and best_state is not None:
            base_model.load_state_dict(best_state)  # type: ignore[arg-type]
            model.eval()
            with torch.no_grad():
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    val_logits = model(x_val_t)
                if multi_head is not None and primary_horizon is not None and y_val_t is not None:
                    best_val = multi_head_val_loss_torch(
                        val_logits,
                        y_val_t,
                        multi_head,
                        primary_horizon=primary_horizon,
                    )
                elif is_classification:
                    best_val = float(
                        nn.functional.binary_cross_entropy_with_logits(
                            val_logits.float(),
                            y_val_t,
                            reduction="mean",
                        )
                    )
                else:
                    val_preds = val_logits.float().cpu().numpy().ravel()
                    best_val = float(mean_squared_error(y_val, val_preds))
        elif best_val == float("inf"):
            best_val = 0.0

        if getattr(device, "type", "") == "cuda":
            torch.cuda.synchronize(device)

        best_state_cpu = {
            k: v.detach().cpu().clone() for k, v in best_state.items()
        }

    return TrainResult(
        val_loss=best_val,
        best_epoch=best_epoch,
        state_dict=best_state_cpu,
        scaler=scaler,
    )


def build_inference_model(
    input_dim: int,
    genome: NNGenome,
    state_dict: dict[str, object],
    config: ModelConfig,
    *,
    task: str,
    device=None,
    n_outputs: int = 1,
):
    """Build a ready-to-eval MLP from saved weights (reuse across predict calls)."""
    torch, _ = _import_torch()
    if device is None:
        device = resolve_device(config)
    model = build_mlp(input_dim, genome, task=task, n_outputs=n_outputs).to(device)
    model.load_state_dict(state_dict)  # type: ignore[arg-type]
    model.eval()
    return model


def _forward_scaled(
    model,
    x_scaled: np.ndarray,
    config: ModelConfig,
    device,
    *,
    task: str,
    multi_head: MultiHeadSpec | None = None,
    primary_horizon: int | None = None,
):
    """Forward an already-scaled matrix through a prebuilt eval model."""
    torch, _ = _import_torch()
    use_amp = bool(config.nn.mixed_precision and getattr(device, "type", "") == "cuda")
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", enabled=use_amp):
            logits = model(torch.from_numpy(x_scaled).float().to(device))
        logits_np = logits.cpu().numpy()
        if multi_head is not None and primary_horizon is not None:
            idx = multi_head.direction_index(primary_horizon)
            return 1.0 / (1.0 + np.exp(-logits_np[:, idx]))
        if task == "classification":
            return torch.sigmoid(logits).cpu().numpy().ravel()
        return logits_np.ravel()


def predict_genome(
    x: np.ndarray,
    genome: NNGenome,
    state_dict: dict[str, object],
    scaler: StandardScaler,
    config: ModelConfig,
    *,
    task: str,
    model=None,
    multi_head: MultiHeadSpec | None = None,
    primary_horizon: int | None = None,
    return_logits: bool = False,
) -> np.ndarray:
    """Run a trained genome forward and return probabilities or regression outputs.

    Pass a prebuilt ``model`` (from :func:`build_inference_model`) to skip rebuilding the
    network and reloading weights on every call (live/run loop, permutation importance).
    """
    torch, _ = _import_torch()
    device = resolve_device(config)
    x_scaled = scaler.transform(x)
    n_outputs = multi_head.n_outputs if multi_head is not None else 1
    if model is None:
        model = build_inference_model(
            x_scaled.shape[1],
            genome,
            state_dict,
            config,
            task=task,
            device=device,
            n_outputs=n_outputs,
        )
    if return_logits and multi_head is not None:
        use_amp = bool(config.nn.mixed_precision and getattr(device, "type", "") == "cuda")
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = model(torch.from_numpy(x_scaled).float().to(device))
            return logits.cpu().numpy()
    return _forward_scaled(
        model,
        x_scaled,
        config,
        device,
        task=task,
        multi_head=multi_head,
        primary_horizon=primary_horizon,
    )


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
    model=None,
    multi_head: MultiHeadSpec | None = None,
    primary_horizon: int | None = None,
) -> dict[str, float]:
    """Estimate feature importance by shuffling each column on a validation slice."""
    if task != "classification":
        return {name: 0.0 for name in feature_names}

    rng = rng or np.random.default_rng(0)
    device = resolve_device(config)
    # Human: build the network and scale X once; shuffling a column then standardising is
    #        identical to standardising then shuffling (per-column stats are unchanged),
    #        so we permute the already-scaled matrix and reuse one eval model per feature.
    # Agent: CALLS build_inference_model once; AVOIDS per-feature rebuild + rescale.
    if model is None:
        model = build_inference_model(
            x.shape[1], genome, state_dict, config, task=task, device=device
        )
    x_scaled = scaler.transform(x)
    baseline_probs = _forward_scaled(
        model,
        x_scaled,
        config,
        device,
        task=task,
        multi_head=multi_head,
        primary_horizon=primary_horizon,
    )
    baseline = float(log_loss(y, baseline_probs, labels=[0, 1]))

    names = feature_names
    if len(names) > max_features:
        idx = rng.choice(len(names), size=max_features, replace=False)
        names = [feature_names[i] for i in sorted(idx)]

    importances: dict[str, float] = {}
    for name in names:
        col_idx = feature_names.index(name)
        x_perm = x_scaled.copy()
        rng.shuffle(x_perm[:, col_idx])
        perm_probs = _forward_scaled(
            model,
            x_perm,
            config,
            device,
            task=task,
            multi_head=multi_head,
            primary_horizon=primary_horizon,
        )
        perm_loss = float(log_loss(y, perm_probs, labels=[0, 1]))
        importances[name] = max(0.0, perm_loss - baseline)

    for name in feature_names:
        importances.setdefault(name, 0.0)
    return importances

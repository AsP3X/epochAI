"""Causal Temporal Convolutional Network backend for walk-forward prediction.

Unlike the dense ``evolved_nn`` MLP, which sees one engineered feature row at a time,
the TCN consumes a sliding window of the last ``lookback`` feature rows and learns
temporal structure directly. Windows are built **causally** (a bar at time ``t`` uses
only rows ``<= t``), so walk-forward integrity is preserved.

The network is a stack of dilated, residual 1D-convolution blocks (dilation ``2**i``
per block) with left-only padding, so the receptive field grows exponentially while
each output position depends solely on current and past bars. It shares the multi-head
output layout (per horizon: direction logit + return quantiles), losses, and
probability calibration with ``evolved_nn``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from epoch_ai.config.settings import ModelConfig, PredictionConfig
from epoch_ai.models.base import MultiHeadModel
from epoch_ai.models.calibration import (
    MultiHeadCalibrator,
    ProbabilityCalibrator,
    load_calibrator_sidecar,
)
from epoch_ai.models.multi_head import (
    MultiHeadSpec,
    multi_head_train_loss,
    multi_head_val_loss_torch,
    parse_structured_predictions,
    targets_to_matrix,
)
from epoch_ai.models.nn_trainer import configure_cuda_runtime, resolve_device
from epoch_ai.utils.logging import get_logger

logger = get_logger(__name__)

CALIBRATION_SUFFIX = ".calibration.json"
SCALER_SUFFIX = ".scaler.json"
ARCH_SUFFIX = ".tcn.json"


def _require_torch():
    try:
        import torch  # noqa: PLC0415

        torch.set_num_threads(1)
    except ImportError as exc:  # pragma: no cover - exercised only without torch
        raise ImportError(
            "model.backend='tcn' requires PyTorch. "
            "Install with `pip install torch` or `pip install -r requirements-optional.txt`."
        ) from exc
    return torch


def _build_network(n_features: int, cfg, n_outputs: int):
    """Construct the dilated causal TCN module (lazy torch import)."""
    _require_torch()
    import torch.nn as nn  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    class _TemporalBlock(nn.Module):
        """Two dilated causal convolutions with a residual connection."""

        def __init__(self, in_ch: int, out_ch: int, k: int, dilation: int, dropout: float):
            super().__init__()
            self.pad = (k - 1) * dilation
            self.conv1 = nn.Conv1d(in_ch, out_ch, k, dilation=dilation)
            self.conv2 = nn.Conv1d(out_ch, out_ch, k, dilation=dilation)
            self.dropout = nn.Dropout(dropout)
            self.relu = nn.ReLU()
            self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

        def forward(self, x):  # x: (B, C, L)
            out = F.pad(x, (self.pad, 0))
            out = self.dropout(self.relu(self.conv1(out)))
            out = F.pad(out, (self.pad, 0))
            out = self.dropout(self.relu(self.conv2(out)))
            res = x if self.downsample is None else self.downsample(x)
            return self.relu(out + res)

    class _TCNNet(nn.Module):
        def __init__(self):
            super().__init__()
            blocks: list = []
            in_ch = n_features
            for i, out_ch in enumerate(cfg.channels):
                blocks.append(
                    _TemporalBlock(in_ch, int(out_ch), cfg.kernel_size, 2**i, cfg.dropout)
                )
                in_ch = int(out_ch)
            self.network = nn.Sequential(*blocks)
            self.head = nn.Linear(in_ch, n_outputs)

        # Human: split the pre-head trunk activation into its own method so callers can
        #        reuse the shared temporal embedding without the head. ``forward`` still
        #        composes embed + head, so its output stays numerically identical.
        # Agent: embed RETURNS (B, channels[-1]) == last causal step; forward = head(embed).
        def embed(self, x):  # x: (B, C=F, L) -> (B, channels[-1])
            h = self.network(x)
            return h[:, :, -1]

        def forward(self, x):  # x: (B, C=F, L)
            return self.head(self.embed(x))

    return _TCNNet()


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid (avoids exp overflow on large |logits|)."""
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))


def _scale_pos_weight(labels: np.ndarray) -> float:
    n_pos = float((labels > 0.5).sum())
    n_neg = float((labels <= 0.5).sum())
    if n_pos <= 0.0 or n_neg <= 0.0:
        return 1.0
    return n_neg / n_pos


class TCNModel(MultiHeadModel):
    """Dilated causal TCN over windows of engineered features (multi-horizon)."""

    BACKEND = "tcn"
    MODEL_FILENAME = "model.pt"

    def __init__(self, config: ModelConfig, task: str = "classification") -> None:
        self.config = config
        self.task = task
        self.state_dict_: dict[str, object] | None = None
        self.scaler_: object | None = None
        self.feature_names_: list[str] | None = None
        self.best_iteration_: int | None = None
        self.calibrator_: ProbabilityCalibrator | None = None
        self.multi_calibrator_: MultiHeadCalibrator | None = None
        self.multi_head_spec_: MultiHeadSpec | None = None
        self.primary_horizon_: int | None = None
        # Architecture actually used by the trained weights (channels/kernel/dropout/
        # lookback). Captured at fit time and restored on load so the network shape and
        # context length follow the *saved* model, not a drifted live config.
        self.arch_: dict | None = None
        self._importance_cache: pd.Series | None = None
        self._infer_model: object | None = None
        self._infer_device: object | None = None

    # ------------------------------------------------------------- properties
    @property
    def sequence_lookback(self) -> int:
        """Number of past bars (inclusive) each prediction depends on."""
        if self.arch_ is not None:
            return int(self.arch_["lookback"])
        return int(self.config.tcn.lookback)

    @property
    def trunk_dim(self) -> int:
        """Width of the pre-head trunk embedding (== ``channels[-1]``).

        Sourced from the saved ``arch_`` when loaded, else the live config, so it always
        matches the network that actually produces :meth:`embed`.
        """
        return int(self._arch_namespace().channels[-1])

    def _arch_namespace(self) -> SimpleNamespace:
        """Network-shape params (from saved ``arch_`` when present, else live config)."""
        src = self.arch_ if self.arch_ is not None else {
            "channels": list(self.config.tcn.channels),
            "kernel_size": self.config.tcn.kernel_size,
            "dropout": self.config.tcn.dropout,
        }
        return SimpleNamespace(
            channels=list(src["channels"]),
            kernel_size=int(src["kernel_size"]),
            dropout=float(src["dropout"]),
        )

    # --------------------------------------------------------------- windowing
    @staticmethod
    def _gather_windows(x_dev, idx_dev, lookback: int):
        """Build ``(B, F, L)`` causal windows for positions ``idx`` from ``(n, F)``.

        Positions earlier than ``i - lookback + 1`` (before the frame start) are
        zero-padded; post-scaling the feature mean is ~0, so zeros are a neutral fill.
        Only rows ``<= i`` ever contribute, which keeps each window causal.
        """
        torch = _require_torch()
        offsets = torch.arange(-lookback + 1, 1, device=x_dev.device)
        gather = idx_dev[:, None] + offsets[None, :]  # (B, L)
        valid = gather >= 0
        clamped = gather.clamp(min=0)
        windows = x_dev[clamped]  # (B, L, F)
        windows = windows * valid.unsqueeze(-1).to(windows.dtype)
        return windows.permute(0, 2, 1).contiguous()  # (B, F, L)

    # --------------------------------------------------------------------- fit
    def fit(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
        val_fraction: float | None = None,
        *,
        compute_importance: bool | None = None,
        prediction: PredictionConfig | None = None,
        multi_targets: pd.DataFrame | None = None,
        seed_state: dict[str, object] | None = None,
    ) -> TCNModel:
        """Train the TCN with Adam and time-ordered early stopping; optionally calibrate."""
        torch = _require_torch()
        if len(x) != len(y):
            raise ValueError("x and y must have the same length.")
        if prediction is not None:
            self.multi_head_spec_ = MultiHeadSpec.from_prediction(prediction)
            self.primary_horizon_ = prediction.horizon

        self.feature_names_ = list(x.columns)
        self.calibrator_ = None
        self.multi_calibrator_ = None
        self._importance_cache = None
        self._infer_model = None
        self._infer_device = None

        tcn_cfg = self.config.tcn
        self.arch_ = {
            "channels": list(tcn_cfg.channels),
            "kernel_size": int(tcn_cfg.kernel_size),
            "dropout": float(tcn_cfg.dropout),
            "lookback": int(tcn_cfg.lookback),
        }
        if val_fraction is None:
            val_fraction = self.config.val_fraction

        x_arr = x.to_numpy(dtype=np.float64)
        mh = self.multi_head_spec_
        ph = self.primary_horizon_
        if mh is not None and multi_targets is not None:
            if len(multi_targets) != len(x):
                raise ValueError("multi_targets must align with x.")
            y_arr = targets_to_matrix(multi_targets, mh)
        else:
            y_arr = y.to_numpy(dtype=np.float64).reshape(-1, 1)

        has_val = 0.0 < val_fraction < 0.5 and len(x_arr) >= 200
        split = int(len(x_arr) * (1.0 - val_fraction)) if has_val else len(x_arr)

        run_importance = (
            tcn_cfg.compute_importance if compute_importance is None else compute_importance
        )

        device = resolve_device(self.config)
        configure_cuda_runtime(device, self.config)
        n_outputs = mh.n_outputs if mh is not None else 1

        result = self._train(
            x_arr,
            y_arr,
            split=split,
            sample_weight=sample_weight,
            device=device,
            n_outputs=n_outputs,
            multi_head=mh,
            primary_horizon=ph,
            seed_state=seed_state,
            refit_full=self.config.refit_full_after_es,
        )
        self.state_dict_ = result["state_dict"]
        self.scaler_ = result["scaler"]
        self.best_iteration_ = result["best_epoch"]
        logger.info(
            "tcn fit channels=%s lookback=%d val_loss=%.5f",
            list(tcn_cfg.channels),
            self.sequence_lookback,
            result["val_loss"],
        )

        if self.task == "classification" and self.config.calibration != "none" and has_val:
            self._fit_calibration(x_arr, y_arr, split, mh, ph)

        if run_importance and has_val and self.task == "classification":
            self._fit_importance(x_arr, y_arr, split, mh, ph)

        del x_arr, y_arr
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return self

    def _train(
        self,
        x_arr: np.ndarray,
        y_arr: np.ndarray,
        *,
        split: int,
        sample_weight: np.ndarray | None,
        device,
        n_outputs: int,
        multi_head: MultiHeadSpec | None,
        primary_horizon: int | None,
        seed_state: dict[str, object] | None,
        refit_full: bool,
    ) -> dict:
        """Core training loop (scaling, batched windowed SGD, early stopping, refit)."""
        torch = _require_torch()
        import torch.nn as nn  # noqa: PLC0415
        from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

        tcn_cfg = self.config.tcn
        lookback = int(tcn_cfg.lookback)
        is_classification = self.task == "classification"
        if primary_horizon is None and multi_head is not None:
            primary_horizon = multi_head.horizons[-1]

        # Fit the scaler on the train portion only, then transform the whole matrix
        # with those train-only statistics (no future leakage into scaling).
        scaler = StandardScaler()
        scaler.fit(x_arr[:split] if split < len(x_arr) else x_arr)
        x_full_scaled = scaler.transform(x_arr)
        y_f = y_arr.astype(np.float32)
        if y_f.ndim == 1:
            y_f = y_f.reshape(-1, 1)

        x_dev = torch.from_numpy(x_full_scaled).float().to(device)
        y_dev = torch.from_numpy(y_f).float().to(device)
        w_dev = (
            None
            if sample_weight is None
            else torch.from_numpy(sample_weight.astype(np.float32)).to(device)
        )

        pos_weight = None
        if is_classification and self.config.class_weight == "balanced":
            if multi_head is not None:
                pos_weight = [
                    torch.tensor(
                        [_scale_pos_weight(y_f[:split, multi_head.direction_index(h)])],
                        device=device,
                    )
                    for h in multi_head.horizons
                ]
            else:
                pos_weight = torch.tensor([_scale_pos_weight(y_f[:split])], device=device)

        use_amp = bool(tcn_cfg.mixed_precision and device.type == "cuda")
        net = _build_network(x_full_scaled.shape[1], tcn_cfg, n_outputs).to(device)
        if seed_state is not None:
            try:
                net.load_state_dict(seed_state)  # type: ignore[arg-type]
            except (RuntimeError, ValueError):
                pass
        optimizer = torch.optim.Adam(
            net.parameters(),
            lr=tcn_cfg.learning_rate,
            weight_decay=tcn_cfg.weight_decay,
        )
        criterion = None
        if multi_head is None:
            if is_classification:
                criterion = nn.BCEWithLogitsLoss(
                    pos_weight=pos_weight if not isinstance(pos_weight, list) else None,
                    reduction="none",
                )
            else:
                criterion = nn.MSELoss(reduction="none")

        has_val = split < len(x_arr)
        train_idx = torch.arange(0, split, device=device)
        val_idx = torch.arange(split, len(x_arr), device=device) if has_val else None
        batch = int(tcn_cfg.batch_size)

        def _forward_batch(idx_batch):
            windows = self._gather_windows(x_dev, idx_batch, lookback)
            return net(windows)

        def _val_loss() -> float:
            net.eval()
            logits_parts = []
            with torch.no_grad():
                for start in range(0, len(val_idx), batch):
                    ib = val_idx[start : start + batch]
                    with torch.autocast(device_type="cuda", enabled=use_amp):
                        logits_parts.append(_forward_batch(ib).float())
            logits = torch.cat(logits_parts, dim=0)
            y_val = y_dev[val_idx]
            if multi_head is not None and primary_horizon is not None:
                return multi_head_val_loss_torch(
                    logits, y_val, multi_head, primary_horizon=primary_horizon
                )
            if is_classification:
                return float(
                    nn.functional.binary_cross_entropy_with_logits(
                        logits, y_val, reduction="mean"
                    )
                )
            return float(nn.functional.mse_loss(logits, y_val))

        best_state: dict[str, object] | None = None
        best_val = float("inf")
        best_epoch = 0
        patience_left = tcn_cfg.patience

        for epoch in range(tcn_cfg.max_epochs):
            net.train()
            perm = train_idx[torch.randperm(len(train_idx), device=device)]
            for start in range(0, len(perm), batch):
                ib = perm[start : start + batch]
                if len(ib) < 1:
                    continue
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    logits = _forward_batch(ib)
                    yb = y_dev[ib]
                    if multi_head is not None:
                        loss = multi_head_train_loss(
                            logits,
                            yb,
                            multi_head,
                            pos_weights=pos_weight if isinstance(pos_weight, list) else None,
                        )
                        if w_dev is not None:
                            loss = loss * w_dev[ib].mean()
                    else:
                        loss_vec = criterion(logits, yb)  # type: ignore[misc]
                        if w_dev is not None:
                            loss = (loss_vec.view(-1) * w_dev[ib]).mean()
                        else:
                            loss = loss_vec.mean()
                loss.backward()
                optimizer.step()

            if not has_val:
                if epoch + 1 >= min(tcn_cfg.max_epochs // 4, 30):
                    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                    best_epoch = epoch + 1
                    best_val = 0.0
                    break
                continue

            val_loss = _val_loss()
            if val_loss + 1e-6 < best_val:
                best_val = float(val_loss)
                best_epoch = epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                patience_left = tcn_cfg.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is None:
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            best_epoch = max(1, tcn_cfg.max_epochs // 4)

        # Optional refit on the full window (keep freshest bars) for best_epoch epochs.
        if refit_full and has_val:
            full_scaler = StandardScaler()
            x_refit = full_scaler.fit_transform(x_arr)
            x_dev = torch.from_numpy(x_refit).float().to(device)
            all_idx = torch.arange(0, len(x_arr), device=device)
            net = _build_network(x_refit.shape[1], tcn_cfg, n_outputs).to(device)
            try:
                net.load_state_dict(best_state)  # type: ignore[arg-type]
            except (RuntimeError, ValueError):
                pass
            optimizer = torch.optim.Adam(
                net.parameters(),
                lr=tcn_cfg.learning_rate,
                weight_decay=tcn_cfg.weight_decay,
            )
            for _ in range(best_epoch):
                net.train()
                perm = all_idx[torch.randperm(len(all_idx), device=device)]
                for start in range(0, len(perm), batch):
                    ib = perm[start : start + batch]
                    if len(ib) < 1:
                        continue
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", enabled=use_amp):
                        logits = _forward_batch(ib)
                        yb = y_dev[ib]
                        if multi_head is not None:
                            loss = multi_head_train_loss(
                                logits,
                                yb,
                                multi_head,
                                pos_weights=pos_weight if isinstance(pos_weight, list) else None,
                            )
                            if w_dev is not None:
                                loss = loss * w_dev[ib].mean()
                        else:
                            loss_vec = criterion(logits, yb)  # type: ignore[misc]
                            if w_dev is not None:
                                loss = (loss_vec.view(-1) * w_dev[ib]).mean()
                            else:
                                loss = loss_vec.mean()
                    loss.backward()
                    optimizer.step()
            scaler = full_scaler
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return {
            "state_dict": best_state,
            "scaler": scaler,
            "best_epoch": best_epoch,
            "val_loss": best_val,
        }

    # --------------------------------------------------------------- inference
    def _inference_net(self):
        """Lazily build and cache the eval network for the current device + weights."""
        if self.state_dict_ is None:
            raise RuntimeError("Model is not trained.")
        device = resolve_device(self.config)
        if self._infer_model is None or self._infer_device != device:
            n_features = len(self.feature_names_ or [])
            n_out = self.multi_head_spec_.n_outputs if self.multi_head_spec_ is not None else 1
            net = _build_network(n_features, self._arch_namespace(), n_out).to(device)
            net.load_state_dict(self.state_dict_)  # type: ignore[arg-type]
            net.eval()
            self._infer_model = net
            self._infer_device = device
        return self._infer_model

    def _forward_frame(self, x: pd.DataFrame, *, embedding: bool = False) -> np.ndarray:
        """Return per-row logits (``n_rows x n_outputs``) or the trunk embedding.

        ``x`` should include up to ``sequence_lookback - 1`` leading context rows; the
        caller trims them. Rows whose window extends before the frame are zero-padded.

        When ``embedding`` is False (default) this returns raw head logits, so all
        existing callers (``predict``, ``predict_logits``, ``predict_structured``,
        calibration, importance) keep identical behavior. When True it instead returns
        ``net.embed(windows)`` -> shape ``(n_rows, channels[-1])``, reusing the same
        causal windowing so the embedding stays walk-forward safe.
        """
        torch = _require_torch()
        if self.feature_names_ is not None:
            x = x[self.feature_names_]
        device = resolve_device(self.config)
        net = self._inference_net()
        x_scaled = self.scaler_.transform(x.to_numpy(dtype=np.float64))  # type: ignore[union-attr]
        x_dev = torch.from_numpy(x_scaled).float().to(device)
        lookback = self.sequence_lookback
        batch = int(self.config.tcn.batch_size)
        use_amp = bool(self.config.tcn.mixed_precision and device.type == "cuda")
        idx_all = torch.arange(0, len(x_scaled), device=device)
        parts = []
        with torch.inference_mode():
            for start in range(0, len(idx_all), batch):
                ib = idx_all[start : start + batch]
                windows = self._gather_windows(x_dev, ib, lookback)
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    # Agent: embed path skips the head; forward path is unchanged.
                    out = net.embed(windows) if embedding else net(windows)
                    parts.append(out.float().cpu().numpy())
        return np.concatenate(parts, axis=0)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Return calibrated P(up) for the primary horizon (one value per input row)."""
        logits = self._forward_frame(x)
        mh = self.multi_head_spec_
        if mh is not None and self.primary_horizon_ is not None:
            idx = mh.direction_index(self.primary_horizon_)
            raw = _sigmoid(logits[:, idx])
            if self.multi_calibrator_ is not None:
                return self.multi_calibrator_.transform(self.primary_horizon_, raw)
            return raw
        raw = _sigmoid(logits[:, 0]) if self.task == "classification" else logits[:, 0]
        if self.calibrator_ is not None:
            return self.calibrator_.transform(raw)
        return raw

    def predict_logits(self, x: pd.DataFrame) -> np.ndarray:
        """Return raw multi-head logits (``n_rows x n_outputs``)."""
        if self.multi_head_spec_ is None:
            raise RuntimeError("predict_logits requires a multi-head model.")
        return self._forward_frame(x)

    def predict_structured(self, x: pd.DataFrame) -> dict[int, dict[str, np.ndarray | float]]:
        """Parse multi-head outputs into per-horizon quantile returns and P(up)."""
        if self.multi_head_spec_ is None or self.primary_horizon_ is None:
            raise RuntimeError("predict_structured requires a multi-head model.")
        logits = self.predict_logits(x)
        parsed = parse_structured_predictions(
            logits, self.multi_head_spec_, primary_horizon=self.primary_horizon_
        )
        if self.multi_calibrator_ is not None:
            for h, block in parsed.items():
                if isinstance(block.get("p_up"), np.ndarray):
                    block["p_up"] = self.multi_calibrator_.transform(h, block["p_up"])
        elif self.calibrator_ is not None:
            for h, block in parsed.items():
                if h == self.primary_horizon_ and isinstance(block.get("p_up"), np.ndarray):
                    block["p_up"] = self.calibrator_.transform(block["p_up"])
        return parsed

    def embed(self, x: pd.DataFrame) -> np.ndarray:
        """Return the causal trunk embedding ``(n_rows, trunk_dim)`` for every row.

        ``trunk_dim == channels[-1]`` (from ``arch_`` when loaded). This is the pre-head
        activation the head consumes; it reuses the same causal windowing as prediction,
        so row ``i`` depends only on rows ``<= i``. Unlike :meth:`predict_structured`,
        this does **not** require a multi-head spec -- it works for any trained TCN.
        """
        # Agent: RETURNS (n_rows, channels[-1]); CAUSAL via _gather_windows; head skipped.
        return self._forward_frame(x, embedding=True)

    def seed_payload(self) -> dict:
        """Warm-start the next retrain from this champion's weights."""
        if self.state_dict_ is None:
            return {}
        return {"seed_state": self.state_dict_}

    # ------------------------------------------------------------- calibration
    def _fit_calibration(self, x_arr, y_arr, split, mh, ph) -> None:
        """Fit probability calibration on the held-out validation tail."""
        val_frame = pd.DataFrame(x_arr[split:], columns=self.feature_names_)
        logits = self._forward_frame(val_frame)
        if mh is not None and ph is not None:
            parsed = parse_structured_predictions(logits, mh, primary_horizon=ph)
            raw_by_h = {h: parsed[h]["p_up"] for h in mh.horizons}
            labels_by_h = {h: y_arr[split:, mh.direction_index(h)] for h in mh.horizons}
            self.multi_calibrator_ = MultiHeadCalibrator.fit(
                raw_by_h, labels_by_h, mh.horizons, self.config.calibration
            )
        else:
            raw = _sigmoid(logits[:, 0])
            self.calibrator_ = ProbabilityCalibrator.fit(
                raw, y_arr[split:].ravel(), self.config.calibration
            )

    def _fit_importance(self, x_arr, y_arr, split, mh, ph, max_features: int = 40) -> None:
        """Permutation importance on the validation tail (primary horizon logloss)."""
        from sklearn.metrics import log_loss  # noqa: PLC0415

        names = list(self.feature_names_ or [])
        if not names:
            return
        rng = np.random.default_rng(0)
        val = x_arr[split:].copy()
        if mh is not None and ph is not None:
            y_imp = y_arr[split:, mh.direction_index(ph)]
        else:
            y_imp = y_arr[split:].ravel()

        def _probs(arr) -> np.ndarray:
            frame = pd.DataFrame(arr, columns=names)
            logits = self._forward_frame(frame)
            idx = mh.direction_index(ph) if (mh is not None and ph is not None) else 0
            return _sigmoid(logits[:, idx])

        baseline = float(log_loss(y_imp, _probs(val), labels=[0, 1]))
        sel = names
        if len(names) > max_features:
            pick = rng.choice(len(names), size=max_features, replace=False)
            sel = [names[i] for i in sorted(pick)]
        importances: dict[str, float] = {}
        for name in sel:
            col = names.index(name)
            perm = val.copy()
            rng.shuffle(perm[:, col])
            importances[name] = max(0.0, float(log_loss(y_imp, _probs(perm), labels=[0, 1])) - baseline)
        for name in names:
            importances.setdefault(name, 0.0)
        self._importance_cache = pd.Series(importances, name="permutation").sort_values(
            ascending=False
        )

    def feature_importance(self) -> pd.Series:
        """Return cached permutation importances (zeros when unavailable)."""
        if self._importance_cache is not None:
            return self._importance_cache
        if self.feature_names_ is None:
            raise RuntimeError("Model is not trained.")
        return pd.Series(0.0, index=self.feature_names_, name="permutation")

    # ------------------------------------------------------------------- io
    def save(self, path: str) -> None:
        """Persist weights, architecture, scaler and optional calibration sidecars."""
        if self.state_dict_ is None or self.scaler_ is None:
            raise RuntimeError("Cannot save an untrained model.")
        torch = _require_torch()
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "state_dict": self.state_dict_,
            "task": self.task,
            "best_epoch": self.best_iteration_,
            "feature_names": self.feature_names_,
            "multi_head": self.multi_head_spec_.to_dict() if self.multi_head_spec_ else None,
            "primary_horizon": self.primary_horizon_,
        }
        torch.save(payload, path, _use_new_zipfile_serialization=True)

        arch = self.arch_ or {
            "channels": list(self.config.tcn.channels),
            "kernel_size": self.config.tcn.kernel_size,
            "dropout": self.config.tcn.dropout,
            "lookback": self.config.tcn.lookback,
        }
        path_obj.with_name(path_obj.name + ARCH_SUFFIX).write_text(
            json.dumps(arch, separators=(",", ":")), encoding="utf-8"
        )
        scaler = self.scaler_
        path_obj.with_name(path_obj.name + SCALER_SUFFIX).write_text(
            json.dumps(
                {
                    "mean": scaler.mean_.tolist(),
                    "scale": scaler.scale_.tolist(),
                    "feature_names": self.feature_names_,
                },
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )

        sidecar = path_obj.with_name(path_obj.name + CALIBRATION_SUFFIX)
        cal_payload = None
        if self.multi_calibrator_ is not None:
            cal_payload = self.multi_calibrator_.to_dict()
        elif self.calibrator_ is not None:
            cal_payload = self.calibrator_.to_dict()
        if cal_payload is not None:
            sidecar.write_text(json.dumps(cal_payload, separators=(",", ":")), encoding="utf-8")
        elif sidecar.exists():
            sidecar.unlink()

    @classmethod
    def load(cls, path: str, config: ModelConfig, task: str = "classification") -> TCNModel:
        """Load a saved TCN and its sidecars."""
        from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

        torch = _require_torch()
        model = cls(config, task=task)
        path_obj = Path(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model.state_dict_ = payload["state_dict"]
        model.best_iteration_ = payload.get("best_epoch")
        model.feature_names_ = list(payload.get("feature_names") or [])

        # Restore the architecture the weights were trained with (channels/kernel/
        # dropout/lookback) so the rebuilt network matches even if the live config drifted.
        arch_path = path_obj.with_name(path_obj.name + ARCH_SUFFIX)
        if arch_path.exists():
            arch = json.loads(arch_path.read_text(encoding="utf-8"))
            model.arch_ = {
                "channels": list(arch["channels"]),
                "kernel_size": int(arch["kernel_size"]),
                "dropout": float(arch["dropout"]),
                "lookback": int(arch["lookback"]),
            }

        mh_payload = payload.get("multi_head")
        if mh_payload:
            model.multi_head_spec_ = MultiHeadSpec.from_dict(mh_payload)
            ph = payload.get("primary_horizon")
            model.primary_horizon_ = (
                int(ph) if ph is not None else model.multi_head_spec_.horizons[-1]
            )

        scaler_path = path_obj.with_name(path_obj.name + SCALER_SUFFIX)
        scaler_payload = json.loads(scaler_path.read_text(encoding="utf-8"))
        scaler = StandardScaler()
        scaler.mean_ = np.asarray(scaler_payload["mean"], dtype=np.float64)
        scaler.scale_ = np.asarray(scaler_payload["scale"], dtype=np.float64)
        scaler.n_features_in_ = len(scaler.mean_)
        model.scaler_ = scaler
        if not model.feature_names_:
            model.feature_names_ = list(scaler_payload.get("feature_names") or [])

        sidecar = path_obj.with_name(path_obj.name + CALIBRATION_SUFFIX)
        if sidecar.exists():
            loaded = load_calibrator_sidecar(json.loads(sidecar.read_text(encoding="utf-8")))
            if isinstance(loaded, MultiHeadCalibrator):
                model.multi_calibrator_ = loaded
            else:
                model.calibrator_ = loaded
        return model

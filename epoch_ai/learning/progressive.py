"""Progressive historical learning engine (the heart of the system).

This implements an **expanding-window / progressive walk-forward** loop:

    1. Start with the *oldest* available data up to an initial cutoff.
    2. Train the model.
    3. Predict the next forward step of unseen candles.
    4. Collect the realised outcomes + rich influencing context for that step.
    5. Append the new samples to the training set (optionally recency-weighted).
    6. Retrain (every ``retrain_frequency`` steps) and advance the window.
    7. Repeat across the entire history.

Every prediction (with its full feature vector) and every outcome (with context) is
optionally persisted to the :class:`~epoch_ai.logging_system.store.PredictionStore`,
so the exact same mechanism powers both the backtest simulation and the live
retraining job. The per-step out-of-sample metrics let us *measure how the model
improves as it walks through history*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from epoch_ai.config.settings import AppConfig
from epoch_ai.execution.risk import RiskManager
from epoch_ai.execution.safety import SafetyScorer
from epoch_ai.features.pipeline import build_multi_horizon_targets, build_target, forward_return
from epoch_ai.learning.checkpoint import (
    build_checkpoint,
    clear_checkpoint,
    load_checkpoint,
    resolve_checkpoint_path,
    save_checkpoint,
    validate_checkpoint,
)
from epoch_ai.learning.step_metrics import (
    classification_step_metrics,
    multi_horizon_classification_step_metrics,
    regression_step_metrics,
)
from epoch_ai.learning.weighting import recency_weights
from epoch_ai.logging_system.multi_horizon_log import log_immediate_outcomes
from epoch_ai.logging_system.schemas import OutcomeLog, PredictionLog
from epoch_ai.logging_system.store import PredictionStore
from epoch_ai.models.base import BaseModel
from epoch_ai.models.evolved_nn_model import EvolvedNNModel
from epoch_ai.models.factory import build_model
from epoch_ai.models.nn_genome import NNGenome
from epoch_ai.models.registry import ModelRegistry
from epoch_ai.services.types import build_multi_horizon_from_structured
from epoch_ai.utils.logging import get_logger
from epoch_ai.utils.timeframe import timeframe_to_minutes

logger = get_logger(__name__)


@dataclass(slots=True)
class ProgressiveResult:
    """Outputs of a progressive walk-forward run.

    Attributes:
        predictions: Per-bar out-of-sample predictions across the walk-forward
            region (index = entry timestamp).
        step_history: Per-step diagnostics (train size, OOS accuracy, logloss).
        feature_importance: Importances from the final trained model.
        final_model_version: Registry label of the final model (if registered).
    """

    predictions: pd.DataFrame
    step_history: pd.DataFrame
    feature_importance: pd.Series = field(default_factory=pd.Series)
    final_model_version: str | None = None
    resumed_from_step: int | None = None


class ProgressiveLearningEngine:
    """Run expanding-window walk-forward training and prediction."""

    def __init__(self, config: AppConfig, register_models: bool = False) -> None:
        self.config = config
        self.risk_manager = RiskManager(config.risk, config.prediction, config.safety)
        self._safety_scorer = SafetyScorer(config.safety) if config.safety.enabled else None
        self.register_models = register_models
        self.registry = ModelRegistry(config.model.model_dir) if register_models else None

    # ------------------------------------------------------------------- helpers
    def _sample_weights(self, n: int) -> np.ndarray | None:
        """Recency-decayed sample weights for the current training set."""
        # Agent: CALLS recency_weights; CAUSAL weights depend only on row age (oldest-first).
        return recency_weights(n, self.config.walk_forward.recency_half_life)

    def _context(self, market: pd.DataFrame, entry_pos: int, horizon: int) -> dict[str, float]:
        """Capture rich influencing context realised during the holding period."""
        window = market.iloc[entry_pos + 1 : entry_pos + 1 + horizon]
        if window.empty:
            return {}
        entry_close = float(market["close"].iloc[entry_pos])
        path = window["close"].to_numpy()
        context: dict[str, float] = {
            "mfe": float(path.max() / entry_close - 1.0),
            "mae": float(path.min() / entry_close - 1.0),
            "realized_vol": float(pd.Series(path).pct_change().std(ddof=0) or 0.0),
        }
        if "volume" in window:
            base = market["volume"].iloc[max(0, entry_pos - 48) : entry_pos + 1].mean()
            context["volume_spike"] = float(window["volume"].max() / base) if base else 0.0
        if "funding_rate" in window:
            context["funding_shift"] = float(
                window["funding_rate"].iloc[-1] - window["funding_rate"].iloc[0]
            )
        if "liquidations" in window:
            context["max_liquidation"] = float(window["liquidations"].max())
        return context

    def _persist_checkpoint(
        self,
        path: Path,
        *,
        step_idx: int,
        cutoff: int,
        model_version: str | None,
        n_features: int,
        resolved_rows: int,
    ) -> None:
        """Save or clear the walk-forward checkpoint after a completed step."""
        wf = self.config.walk_forward
        if not wf.checkpoint_enabled:
            return
        if cutoff >= resolved_rows:
            clear_checkpoint(path)
            return
        save_checkpoint(
            path,
            build_checkpoint(
                step_idx=step_idx,
                cutoff=cutoff,
                model_version=model_version if self.register_models else None,
                config=self.config,
                n_features=n_features,
                resolved_rows=resolved_rows,
            ),
        )

    # ----------------------------------------------------------------------- run
    def run(
        self,
        market: pd.DataFrame,
        features: pd.DataFrame,
        store: PredictionStore | None = None,
        *,
        resume: bool = False,
        fresh: bool = False,
    ) -> ProgressiveResult:
        """Execute the full progressive walk-forward simulation.

        Args:
            market: Cleaned OHLCV(+context) frame indexed by ``timestamp``.
            features: Engineered feature matrix (subset of ``market``'s index).
            store: Optional log store; predictions/outcomes are persisted if given.
            resume: When ``True`` and a checkpoint exists, continue from the saved step.
            fresh: Delete any checkpoint before starting (always begins at step 0).

        Returns:
            A :class:`ProgressiveResult`.
        """
        wf = self.config.walk_forward
        horizon = self.config.prediction.horizon
        symbol = self.config.primary_symbol
        # Purge gap: multi-horizon targets look up to ``max_horizon`` bars ahead, so the
        # final ``embargo`` training labels would otherwise overlap (and leak) the
        # prediction window. Dropping them keeps the train/test boundary causally clean.
        embargo = self.config.prediction.resolved_embargo(wf.embargo)

        # Align features with targets/outcomes and keep only resolved rows.
        y = build_target(market, self.config.prediction)
        fwd = forward_return(market, horizon)
        multi = build_multi_horizon_targets(market, self.config.prediction)
        data = features.join(y).join(fwd).join(multi)
        data = data.join(market["close"].rename("close"))
        drop_cols = ["target", "forward_return"]
        for h in self.config.prediction.horizons:
            drop_cols.extend([f"ret_{h}", f"target_{h}"])
        data = data.dropna(subset=drop_cols)
        feature_cols = list(features.columns)

        n = len(data)
        if n <= wf.initial_train_period + 1:
            raise ValueError(
                f"Not enough resolved rows ({n}) for initial_train_period="
                f"{wf.initial_train_period}. Use more data or a smaller window."
            )

        # Positional view of the raw market frame for context lookups.
        market_pos = {ts: i for i, ts in enumerate(market.index)}

        x_all = data[feature_cols]
        y_all = data["target"]
        fwd_all = data["forward_return"]
        ts_all = data.index

        checkpoint_path = resolve_checkpoint_path(self.config)
        if fresh:
            clear_checkpoint(checkpoint_path)

        model: BaseModel | None = None
        model_version = "untrained"
        pred_records: list[dict] = []
        step_records: list[dict] = []
        resumed_from_step: int | None = None

        cutoff = wf.initial_train_period
        step_idx = 0
        if resume and wf.checkpoint_enabled:
            state = load_checkpoint(checkpoint_path)
            if state is not None:
                if state.completed:
                    clear_checkpoint(checkpoint_path)
                else:
                    validate_checkpoint(
                        state,
                        self.config,
                        len(feature_cols),
                        n,
                    )
                    step_idx = state.step_idx
                    cutoff = state.cutoff
                    resumed_from_step = step_idx
                    if state.model_version and self.registry is not None:
                        try:
                            model, _ = self.registry.load(
                                state.model_version,
                                self.config.model,
                                task=self.config.prediction.task,
                            )
                            model_version = state.model_version
                        except FileNotFoundError:
                            logger.warning(
                                "Checkpoint model %s missing from registry; will retrain.",
                                state.model_version,
                            )
                            model = None
                    elif state.model_version:
                        logger.warning(
                            "Checkpoint references model %s but register_models=False; "
                            "will retrain.",
                            state.model_version,
                        )
                    logger.info(
                        "Resuming walk-forward from step %d (cutoff=%d, rows=%d, model=%s).",
                        step_idx,
                        cutoff,
                        n,
                        model_version,
                    )

        while cutoff < n:
            if wf.max_steps is not None and step_idx >= wf.max_steps:
                break

            train_start = 0 if wf.expanding else max(0, cutoff - wf.initial_train_period)
            # Exclude the embargo gap so no training label looks into the test window.
            train_end = max(train_start, cutoff - embargo)
            n_train = train_end - train_start
            need_retrain = model is None or (step_idx % wf.retrain_frequency == 0)

            if need_retrain:
                if n_train < 1:
                    raise ValueError(
                        f"Embargo ({embargo}) leaves no training rows at step {step_idx}; "
                        "reduce walk_forward.embargo or increase initial_train_period."
                    )
                x_train = x_all.iloc[train_start:train_end]
                y_train = y_all.iloc[train_start:train_end]
                weights = self._sample_weights(len(x_train))
                # Human: warm-start evolved_nn from the prior champion genome/weights.
                # Agent: READS model.genome_/state_dict_; only for EvolvedNNModel retrain.
                seed_genome: NNGenome | None = None
                seed_state: dict[str, object] | None = None
                if isinstance(model, EvolvedNNModel) and model.genome_ is not None:
                    seed_genome = model.genome_
                    seed_state = model.state_dict_
                test_end = min(cutoff + wf.step_size, n)
                is_final_retrain = test_end >= n or (
                    wf.max_steps is not None and step_idx + 1 >= wf.max_steps
                )
                compute_importance = (
                    self.config.model.nn.compute_importance and is_final_retrain
                )
                model = build_model(self.config.model, task=self.config.prediction.task)
                if isinstance(model, EvolvedNNModel):
                    multi_train = data.loc[
                        x_train.index,
                        [c for c in data.columns if c.startswith(("ret_", "target_"))],
                    ]
                    model.fit(
                        x_train,
                        y_train,
                        sample_weight=weights,
                        compute_importance=compute_importance,
                        seed_genome=seed_genome,
                        seed_state=seed_state,
                        prediction=self.config.prediction,
                        multi_targets=multi_train,
                    )
                else:
                    model.fit(x_train, y_train, sample_weight=weights)
                if self.registry is not None:
                    protect_labels: set[str] = set()
                    if wf.checkpoint_enabled:
                        ckpt = load_checkpoint(checkpoint_path)
                        if ckpt is not None and ckpt.model_version:
                            protect_labels.add(ckpt.model_version)
                    model_version = self.registry.save(
                        model,
                        metadata={
                            "train_start": str(ts_all[train_start]),
                            "train_end": str(ts_all[train_end - 1]),
                            "train_rows": len(x_train),
                            "step": step_idx,
                            "horizons": list(self.config.prediction.horizons),
                            "quantiles": list(self.config.prediction.quantiles),
                            "n_outputs": self.config.prediction.n_outputs,
                        },
                        retain_versions=self.config.model.retain_versions,
                        protect=protect_labels,
                    )
                else:
                    model_version = f"step_{step_idx}"

            test_end = min(cutoff + wf.step_size, n)
            x_test = x_all.iloc[cutoff:test_end]
            preds = model.predict(x_test)
            is_classification = self.config.prediction.task == "classification"
            multi_head = (
                isinstance(model, EvolvedNNModel) and model.multi_head_spec_ is not None
            )
            structured_batch = (
                model.predict_structured(x_test)
                if store is not None and multi_head
                else None
            )
            bar_minutes = timeframe_to_minutes(self.config.timeframe)

            # Collect per-bar arrays for honest out-of-sample step metrics.
            step_preds: list[float] = []
            step_labels: list[int] = []
            step_returns: list[float] = []
            for offset, raw_pred in enumerate(preds):
                pos = cutoff + offset
                ts = ts_all[pos]
                entry_price = float(data["close"].iloc[pos])
                realized_ret = float(fwd_all.iloc[pos])
                # Align the logged label with the pre-built training target instead of
                # recomputing the threshold rule (single source of truth, no drift).
                if is_classification:
                    realized_label = int(y_all.iloc[pos])
                else:
                    realized_label = int(realized_ret > self.config.prediction.threshold)

                feat_row = x_test.iloc[offset]
                safety_assess = (
                    self._safety_scorer.assess(feat_row) if self._safety_scorer else None
                )
                decision = self.risk_manager.decide(
                    float(raw_pred),
                    safety=safety_assess,
                )
                pred_records.append(
                    {
                        "timestamp": ts,
                        "prediction": float(raw_pred),
                        "confidence": decision.confidence,
                        "signal": decision.signal,
                        "target_weight": decision.target_weight,
                        "forward_return": realized_ret,
                        "realized_label": realized_label,
                        "model_version": model_version,
                    }
                )
                step_preds.append(float(raw_pred))
                step_labels.append(realized_label)
                step_returns.append(realized_ret)

                # Persist prediction + outcome (+ context) to the store.
                if store is not None:
                    entry_market_pos = market_pos.get(ts)
                    context = (
                        self._context(market, entry_market_pos, horizon)
                        if entry_market_pos is not None
                        else {}
                    )
                    feature_dict = {
                        k: float(v) for k, v in x_test.iloc[offset].to_dict().items()
                    }
                    if structured_batch is not None and isinstance(model, EvolvedNNModel):
                        spec = model.multi_head_spec_
                        assert spec is not None
                        multi = build_multi_horizon_from_structured(
                            structured_batch,
                            offset,
                            as_of=pd.Timestamp(ts),
                            last_close=entry_price,
                            model_version=model_version,
                            symbol=symbol,
                            timeframe=self.config.timeframe,
                            horizons=list(spec.horizons),
                            horizon_label_fn=self.config.prediction.horizon_label,
                            bar_minutes=bar_minutes,
                        )
                        for forecast in multi.horizons:
                            h = forecast.horizon
                            ret_h = float(data[f"ret_{h}"].iloc[pos])
                            label_h = int(data[f"target_{h}"].iloc[pos])
                            resolve_idx = min(pos + h, n - 1)
                            resolve_ts = ts_all[resolve_idx]
                            band_features = {
                                **feature_dict,
                                "return_q10": float(math.log(forecast.price_p10 / entry_price)),
                                "return_q50": forecast.exp_return,
                                "return_q90": float(math.log(forecast.price_p90 / entry_price)),
                                "price_p10": forecast.price_p10,
                                "price_p50": forecast.price_p50,
                                "price_p90": forecast.price_p90,
                                "head_confidence": forecast.confidence,
                                "reliable": forecast.reliable,
                            }
                            log_immediate_outcomes(
                                store,
                                timestamp=str(ts),
                                symbol=symbol,
                                model_version=model_version,
                                horizon=h,
                                p_up=forecast.p_up,
                                confidence=forecast.confidence,
                                signal=decision.signal,
                                entry_price=entry_price,
                                features=band_features,
                                forward_return=math.exp(ret_h) - 1.0,
                                realized_label=label_h,
                                resolve_timestamp=str(resolve_ts),
                                exit_price=entry_price * math.exp(ret_h),
                                context=context,
                            )
                    else:
                        pred_id = store.log_prediction(
                            PredictionLog(
                                timestamp=str(ts),
                                symbol=symbol,
                                model_version=model_version,
                                horizon=horizon,
                                prediction=float(raw_pred),
                                confidence=decision.confidence,
                                signal=decision.signal,
                                entry_price=entry_price,
                                features=feature_dict,
                            )
                        )
                        resolve_ts = ts_all[min(pos + horizon, n - 1)]
                        store.log_outcome(
                            OutcomeLog(
                                prediction_id=pred_id,
                                resolve_timestamp=str(resolve_ts),
                                forward_return=realized_ret,
                                realized_label=realized_label,
                                exit_price=entry_price * (1.0 + realized_ret),
                                context=context,
                            )
                        )

            n_step = len(preds)
            if n_step > 0:
                record: dict = {
                    "step": step_idx,
                    "train_end": ts_all[train_end - 1],
                    "train_rows": n_train,
                    "test_rows": n_step,
                }
                # Threshold-aware + probabilistic metrics (classification) or
                # directional/RMSE metrics (regression) for a faithful learning curve.
                if is_classification:
                    record.update(
                        classification_step_metrics(
                            np.asarray(step_preds),
                            np.asarray(step_labels),
                            long_threshold=self.config.risk.long_threshold,
                            short_threshold=self.config.risk.short_threshold,
                        )
                    )
                    if (
                        isinstance(model, EvolvedNNModel)
                        and model.multi_head_spec_ is not None
                    ):
                        structured = model.predict_structured(x_test)
                        horizons = model.multi_head_spec_.horizons
                        labels_by_h = {
                            h: data[f"target_{h}"].iloc[cutoff:test_end].to_numpy(dtype=float)
                            for h in horizons
                        }
                        returns_by_h = {
                            h: data[f"ret_{h}"].iloc[cutoff:test_end].to_numpy(dtype=float)
                            for h in horizons
                        }
                        record.update(
                            multi_horizon_classification_step_metrics(
                                structured,
                                labels_by_h,
                                returns_by_h,
                                long_threshold=self.config.risk.long_threshold,
                                short_threshold=self.config.risk.short_threshold,
                                primary_horizon=self.config.prediction.horizon,
                            )
                        )
                    # Human: Track label balance and mean P(up) per step to explain
                    #        first-vs-second-half OOS accuracy drift in backtest reports.
                    # Agent: WRITES test_label_rate, mean_prediction; CAUSAL step-local stats.
                    record["test_label_rate"] = float(np.mean(step_labels))
                    record["mean_prediction"] = float(np.mean(step_preds))
                    logger.info(
                        "Step %d | train=%d | test=%d | acc=%.3f logloss=%.4f "
                        "auc=%.3f brier=%.4f dir_acc=%.3f",
                        step_idx,
                        n_train,
                        n_step,
                        record["oos_accuracy"],
                        record["oos_logloss"],
                        record["oos_auc"],
                        record["oos_brier"],
                        record["oos_directional_accuracy"],
                    )
                else:
                    record.update(
                        regression_step_metrics(
                            np.asarray(step_preds), np.asarray(step_returns)
                        )
                    )
                    logger.info(
                        "Step %d | train=%d | test=%d | dir_acc=%.3f rmse=%.5f",
                        step_idx,
                        n_train,
                        n_step,
                        record["oos_accuracy"],
                        record["oos_rmse"],
                    )
                step_records.append(record)

            cutoff = test_end
            step_idx += 1
            registry_version = model_version if self.register_models else None
            self._persist_checkpoint(
                checkpoint_path,
                step_idx=step_idx,
                cutoff=cutoff,
                model_version=registry_version,
                n_features=len(feature_cols),
                resolved_rows=n,
            )

        if pred_records:
            predictions = pd.DataFrame(pred_records).set_index("timestamp")
        else:
            predictions = pd.DataFrame(
                columns=[
                    "prediction",
                    "confidence",
                    "signal",
                    "target_weight",
                    "forward_return",
                    "realized_label",
                    "model_version",
                ]
            )
        step_history = pd.DataFrame(step_records)
        importance = model.feature_importance() if model is not None else pd.Series(dtype=float)
        final_version = model_version if self.register_models else None
        return ProgressiveResult(
            predictions=predictions,
            step_history=step_history,
            feature_importance=importance,
            final_model_version=final_version,
            resumed_from_step=resumed_from_step,
        )

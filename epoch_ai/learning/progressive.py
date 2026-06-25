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

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from epoch_ai.config.settings import AppConfig
from epoch_ai.execution.risk import RiskManager
from epoch_ai.features.pipeline import build_target, forward_return
from epoch_ai.learning.step_metrics import classification_step_metrics, regression_step_metrics
from epoch_ai.learning.weighting import recency_weights
from epoch_ai.logging_system.schemas import OutcomeLog, PredictionLog
from epoch_ai.logging_system.store import PredictionStore
from epoch_ai.models.lightgbm_model import LightGBMModel
from epoch_ai.models.registry import ModelRegistry
from epoch_ai.utils.logging import get_logger

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


class ProgressiveLearningEngine:
    """Run expanding-window walk-forward training and prediction."""

    def __init__(self, config: AppConfig, register_models: bool = False) -> None:
        self.config = config
        self.risk_manager = RiskManager(config.risk, config.prediction)
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

    # ----------------------------------------------------------------------- run
    def run(
        self,
        market: pd.DataFrame,
        features: pd.DataFrame,
        store: PredictionStore | None = None,
    ) -> ProgressiveResult:
        """Execute the full progressive walk-forward simulation.

        Args:
            market: Cleaned OHLCV(+context) frame indexed by ``timestamp``.
            features: Engineered feature matrix (subset of ``market``'s index).
            store: Optional log store; predictions/outcomes are persisted if given.

        Returns:
            A :class:`ProgressiveResult`.
        """
        wf = self.config.walk_forward
        horizon = self.config.prediction.horizon
        symbol = self.config.primary_symbol

        # Align features with targets/outcomes and keep only resolved rows.
        y = build_target(market, self.config.prediction)
        fwd = forward_return(market, horizon)
        data = features.join(y).join(fwd)
        data = data.join(market["close"].rename("close"))
        data = data.dropna(subset=["target", "forward_return"])
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

        model: LightGBMModel | None = None
        model_version = "untrained"
        pred_records: list[dict] = []
        step_records: list[dict] = []

        cutoff = wf.initial_train_period
        step_idx = 0
        while cutoff < n:
            if wf.max_steps is not None and step_idx >= wf.max_steps:
                break

            train_start = 0 if wf.expanding else max(0, cutoff - wf.initial_train_period)
            need_retrain = model is None or (step_idx % wf.retrain_frequency == 0)

            if need_retrain:
                x_train = x_all.iloc[train_start:cutoff]
                y_train = y_all.iloc[train_start:cutoff]
                weights = self._sample_weights(len(x_train))
                model = LightGBMModel(self.config.model, task=self.config.prediction.task)
                model.fit(x_train, y_train, sample_weight=weights)
                if self.registry is not None:
                    model_version = self.registry.save(
                        model,
                        metadata={
                            "train_start": str(ts_all[train_start]),
                            "train_end": str(ts_all[cutoff - 1]),
                            "train_rows": len(x_train),
                            "step": step_idx,
                        },
                    )
                else:
                    model_version = f"step_{step_idx}"

            test_end = min(cutoff + wf.step_size, n)
            x_test = x_all.iloc[cutoff:test_end]
            preds = model.predict(x_test)
            is_classification = self.config.prediction.task == "classification"

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

                decision = self.risk_manager.decide(float(raw_pred))
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
                            features={
                                k: float(v) for k, v in x_test.iloc[offset].to_dict().items()
                            },
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
                    "train_end": ts_all[cutoff - 1],
                    "train_rows": cutoff - train_start,
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
                    logger.info(
                        "Step %d | train=%d | test=%d | acc=%.3f logloss=%.4f "
                        "auc=%.3f brier=%.4f dir_acc=%.3f",
                        step_idx,
                        cutoff - train_start,
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
                        cutoff - train_start,
                        n_step,
                        record["oos_accuracy"],
                        record["oos_rmse"],
                    )
                step_records.append(record)

            cutoff = test_end
            step_idx += 1

        predictions = pd.DataFrame(pred_records).set_index("timestamp")
        step_history = pd.DataFrame(step_records)
        importance = model.feature_importance() if model is not None else pd.Series(dtype=float)
        final_version = model_version if self.register_models else None
        return ProgressiveResult(
            predictions=predictions,
            step_history=step_history,
            feature_importance=importance,
            final_model_version=final_version,
        )

"""Evolved neural-network backend for walk-forward prediction.

An outer evolutionary loop searches MLP architecture hyper-parameters (layer sizes,
dropout, learning rate). Each candidate is trained with Adam on engineered causal
features; fitness is validation logloss on a time-ordered holdout tail.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from epoch_ai.config.settings import ModelConfig, PredictionConfig
from epoch_ai.models.base import BaseModel
from epoch_ai.models.calibration import (
    MultiHeadCalibrator,
    ProbabilityCalibrator,
    load_calibrator_sidecar,
)
from epoch_ai.models.multi_head import (
    MultiHeadSpec,
    parse_structured_predictions,
    targets_to_matrix,
)
from epoch_ai.models.nn_genome import (
    NNGenome,
    default_genome,
    initialize_population,
    initialize_population_from_seed,
    mutate_genome,
)
from epoch_ai.models.nn_trainer import (
    build_inference_model,
    build_training_cache,
    evolution_max_workers,
    permutation_importance,
    predict_genome,
    resolve_device,
    train_genome,
)
from epoch_ai.utils.logging import get_logger

logger = get_logger(__name__)

CALIBRATION_SUFFIX = ".calibration.json"
GENOME_SUFFIX = ".genome.json"
SCALER_SUFFIX = ".scaler.json"


class EvolvedNNModel(BaseModel):
    """MLP classifier/regressor with evolutionary architecture search."""

    BACKEND = "evolved_nn"
    MODEL_FILENAME = "model.pt"

    def __init__(self, config: ModelConfig, task: str = "classification") -> None:
        self.config = config
        self.task = task
        self.genome_: NNGenome | None = None
        self.state_dict_: dict[str, object] | None = None
        self.scaler_: object | None = None
        self.feature_names_: list[str] | None = None
        self.best_iteration_: int | None = None
        self.calibrator_: ProbabilityCalibrator | None = None
        self.multi_calibrator_: MultiHeadCalibrator | None = None
        self.multi_head_spec_: MultiHeadSpec | None = None
        self.primary_horizon_: int | None = None
        self._importance_cache: pd.Series | None = None
        # Cached eval network reused across predict() calls (built lazily, reset on
        # fit/load). Avoids rebuilding + reloading weights every bar in run/live mode.
        self._infer_model: object | None = None
        self._infer_device: object | None = None

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
        val_fraction: float | None = None,
        *,
        compute_importance: bool | None = None,
        seed_genome: NNGenome | None = None,
        seed_state: dict[str, object] | None = None,
        prediction: PredictionConfig | None = None,
        multi_targets: pd.DataFrame | None = None,
    ) -> EvolvedNNModel:
        """Evolve architecture genes, train the best candidate, optionally calibrate."""
        if len(x) != len(y):
            raise ValueError("x and y must have the same length.")
        if prediction is not None:
            self.multi_head_spec_ = MultiHeadSpec.from_prediction(prediction)
            self.primary_horizon_ = prediction.horizon
        elif self.multi_head_spec_ is None:
            self.primary_horizon_ = None

        self.feature_names_ = list(x.columns)
        self.calibrator_ = None
        self.multi_calibrator_ = None
        self._importance_cache = None
        self._infer_model = None
        self._infer_device = None

        if val_fraction is None:
            val_fraction = self.config.val_fraction

        x_arr = x.to_numpy(dtype=np.float64)
        if self.multi_head_spec_ is not None and multi_targets is not None:
            if len(multi_targets) != len(x):
                raise ValueError("multi_targets must align with x.")
            y_arr = targets_to_matrix(multi_targets, self.multi_head_spec_)
        else:
            y_arr = y.to_numpy(dtype=np.float64).reshape(-1, 1)
        has_val = 0.0 < val_fraction < 0.5 and len(x_arr) >= 200
        split = int(len(x_arr) * (1.0 - val_fraction)) if has_val else len(x_arr)

        evolution = self.config.evolution
        nn_cfg = self.config.nn
        rng = np.random.default_rng(evolution.seed)

        run_importance = (
            self.config.nn.compute_importance
            if compute_importance is None
            else compute_importance
        )

        mh = self.multi_head_spec_
        ph = self.primary_horizon_

        cache = build_training_cache(
            x_arr,
            y_arr,
            self.config,
            task=self.task,
            sample_weight=sample_weight,
            val_fraction=val_fraction,
            split=split,
            multi_head=mh,
        )

        def _initial_state(genome: NNGenome) -> dict[str, object] | None:
            if seed_genome is None or seed_state is None:
                return None
            if genome.hidden_sizes != seed_genome.hidden_sizes:
                return None
            if (
                genome.dropout != seed_genome.dropout
                or genome.use_batch_norm != seed_genome.use_batch_norm
            ):
                return None
            return seed_state

        def _train_candidate(genome: NNGenome):
            result = train_genome(
                x_arr,
                y_arr,
                genome,
                self.config,
                task=self.task,
                sample_weight=sample_weight,
                val_fraction=val_fraction,
                split=split,
                refit_full=False,
                cache=cache,
                initial_state=_initial_state(genome),
                multi_head=mh,
                primary_horizon=ph,
            )
            return result.val_loss, genome, result

        if evolution.fast_fit or not evolution.enabled:
            best_genome = seed_genome if seed_genome is not None else default_genome(nn_cfg)
            trained = train_genome(
                x_arr,
                y_arr,
                best_genome,
                self.config,
                task=self.task,
                sample_weight=sample_weight,
                val_fraction=val_fraction,
                split=split,
                refit_full=self.config.refit_full_after_es,
                cache=cache,
                initial_state=seed_state if seed_genome is not None else None,
                multi_head=mh,
                primary_horizon=ph,
            )
            logger.info(
                "evolved_nn fast_fit genome=%s val_loss=%.5f",
                best_genome.hidden_sizes,
                trained.val_loss,
            )
        else:
            if seed_genome is not None:
                population = initialize_population_from_seed(
                    rng,
                    nn_cfg,
                    evolution,
                    seed_genome,
                )
            else:
                population = initialize_population(rng, nn_cfg, evolution)
            winning_genome: NNGenome | None = None
            best_fitness = float("inf")
            best_trained = None
            stale_generations = 0

            max_workers = evolution_max_workers(self.config, evolution.population_size)
            use_parallel = (
                evolution.parallel_candidates
                and max_workers > 1
                and len(population) > 1
            )
            logger.info(
                "evolved_nn evolution: workers=%d population=%d generations=%d parallel=%s",
                max_workers,
                evolution.population_size,
                evolution.generations,
                use_parallel,
            )

            if use_parallel:
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    for generation in range(evolution.generations):
                        scores = list(pool.map(_train_candidate, population))

                        scores.sort(key=lambda item: item[0])
                        gen_best_loss, gen_best_genome, gen_best_result = scores[0]
                        logger.info(
                            "evolved_nn generation=%d best_val_loss=%.5f genome=%s",
                            generation + 1,
                            gen_best_loss,
                            gen_best_genome.hidden_sizes,
                        )
                        if gen_best_loss < best_fitness:
                            best_fitness = gen_best_loss
                            winning_genome = gen_best_genome
                            best_trained = gen_best_result
                            stale_generations = 0
                        else:
                            stale_generations += 1

                        if (
                            evolution.early_stop_patience is not None
                            and stale_generations >= evolution.early_stop_patience
                        ):
                            logger.info(
                                "evolved_nn early stop: %d generations without improvement.",
                                stale_generations,
                            )
                            break

                        elite_n = max(1, int(evolution.population_size * evolution.elite_fraction))
                        elites = [g for _, g, _ in scores[:elite_n]]
                        next_pop: list[NNGenome] = list(elites)
                        while len(next_pop) < evolution.population_size:
                            parent = elites[int(rng.integers(0, len(elites)))]
                            next_pop.append(
                                mutate_genome(
                                    parent,
                                    rng,
                                    nn_cfg,
                                    sigma=evolution.mutation_sigma,
                                )
                            )
                        population = next_pop
            else:
                for generation in range(evolution.generations):
                    scores = [_train_candidate(genome) for genome in population]

                    scores.sort(key=lambda item: item[0])
                    gen_best_loss, gen_best_genome, gen_best_result = scores[0]
                    logger.info(
                        "evolved_nn generation=%d best_val_loss=%.5f genome=%s",
                        generation + 1,
                        gen_best_loss,
                        gen_best_genome.hidden_sizes,
                    )
                    if gen_best_loss < best_fitness:
                        best_fitness = gen_best_loss
                        winning_genome = gen_best_genome
                        best_trained = gen_best_result
                        stale_generations = 0
                    else:
                        stale_generations += 1

                    if (
                        evolution.early_stop_patience is not None
                        and stale_generations >= evolution.early_stop_patience
                    ):
                        logger.info(
                            "evolved_nn early stop: %d generations without improvement.",
                            stale_generations,
                        )
                        break

                    elite_n = max(1, int(evolution.population_size * evolution.elite_fraction))
                    elites = [g for _, g, _ in scores[:elite_n]]
                    next_pop: list[NNGenome] = list(elites)
                    while len(next_pop) < evolution.population_size:
                        parent = elites[int(rng.integers(0, len(elites)))]
                        next_pop.append(
                            mutate_genome(
                                parent,
                                rng,
                                nn_cfg,
                                sigma=evolution.mutation_sigma,
                            )
                        )
                    population = next_pop

            assert winning_genome is not None and best_trained is not None
            best_genome = winning_genome
            trained = train_genome(
                x_arr,
                y_arr,
                best_genome,
                self.config,
                task=self.task,
                sample_weight=sample_weight,
                val_fraction=val_fraction,
                split=split,
                refit_full=self.config.refit_full_after_es,
                cache=cache,
                initial_state=_initial_state(best_genome),
                multi_head=mh,
                primary_horizon=ph,
            )
            if best_trained.val_loss < trained.val_loss:
                trained = best_trained

        self.genome_ = best_genome
        self.state_dict_ = trained.state_dict
        self.scaler_ = trained.scaler
        self.best_iteration_ = trained.best_epoch

        if self.task == "classification" and self.config.calibration != "none" and has_val:
            if mh is not None and ph is not None:
                raw_logits = predict_genome(
                    x_arr[split:],
                    self.genome_,
                    self.state_dict_,
                    self.scaler_,
                    self.config,
                    task=self.task,
                    multi_head=mh,
                    primary_horizon=ph,
                    model=self._inference_model(),
                    return_logits=True,
                )
                parsed = parse_structured_predictions(
                    raw_logits, mh, primary_horizon=ph
                )
                raw_by_h = {h: parsed[h]["p_up"] for h in mh.horizons}
                labels_by_h = {
                    h: y_arr[split:, mh.direction_index(h)] for h in mh.horizons
                }
                self.multi_calibrator_ = MultiHeadCalibrator.fit(
                    raw_by_h,
                    labels_by_h,
                    mh.horizons,
                    self.config.calibration,
                )
            else:
                raw_val = predict_genome(
                    x_arr[split:],
                    self.genome_,
                    self.state_dict_,
                    self.scaler_,
                    self.config,
                    task=self.task,
                )
                y_cal = y_arr[split:].ravel()
                self.calibrator_ = ProbabilityCalibrator.fit(
                    raw_val,
                    y_cal,
                    self.config.calibration,
                )

        if run_importance and has_val and self.task == "classification":
            if mh is not None and ph is not None:
                y_imp = y_arr[split:, mh.direction_index(ph)]
            else:
                y_imp = y_arr[split:].ravel()
            imp = permutation_importance(
                x_arr[split:],
                y_imp,
                self.genome_,
                self.state_dict_,
                self.scaler_,
                self.config,
                task=self.task,
                feature_names=self.feature_names_,
                rng=rng,
                model=self._inference_model(),
                multi_head=mh,
                primary_horizon=ph,
            )
            self._importance_cache = pd.Series(imp, name="permutation").sort_values(
                ascending=False
            )

        return self

    def _inference_model(self):
        """Lazily build and cache the eval network for the current device + weights."""
        if self.genome_ is None or self.state_dict_ is None:
            raise RuntimeError("Model is not trained.")
        device = resolve_device(self.config)
        if self._infer_model is None or self._infer_device != device:
            input_dim = len(self.feature_names_ or [])
            n_out = self.multi_head_spec_.n_outputs if self.multi_head_spec_ is not None else 1
            self._infer_model = build_inference_model(
                input_dim,
                self.genome_,
                self.state_dict_,
                self.config,
                task=self.task,
                device=device,
                n_outputs=n_out,
            )
            self._infer_device = device
        return self._infer_model

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Return calibrated probabilities (classification) or raw regression outputs."""
        if self.genome_ is None or self.state_dict_ is None or self.scaler_ is None:
            raise RuntimeError("Model is not trained. Call fit() first.")
        if self.feature_names_ is not None:
            x = x[self.feature_names_]
        raw = predict_genome(
            x.to_numpy(dtype=np.float64),
            self.genome_,
            self.state_dict_,
            self.scaler_,
            self.config,
            task=self.task,
            model=self._inference_model(),
            multi_head=self.multi_head_spec_,
            primary_horizon=self.primary_horizon_,
        )
        if self.multi_calibrator_ is not None and self.primary_horizon_ is not None:
            return self.multi_calibrator_.transform(self.primary_horizon_, raw)
        if self.calibrator_ is not None:
            return self.calibrator_.transform(raw)
        return raw

    def predict_logits(self, x: pd.DataFrame) -> np.ndarray:
        """Return raw multi-head logits (``n_rows x n_outputs``) when trained multi-horizon."""
        if self.multi_head_spec_ is None:
            raise RuntimeError("predict_logits requires a multi-head model.")
        if self.feature_names_ is not None:
            x = x[self.feature_names_]
        return predict_genome(
            x.to_numpy(dtype=np.float64),
            self.genome_,
            self.state_dict_,
            self.scaler_,
            self.config,
            task=self.task,
            model=self._inference_model(),
            multi_head=self.multi_head_spec_,
            primary_horizon=self.primary_horizon_,
            return_logits=True,
        )

    def predict_structured(self, x: pd.DataFrame) -> dict[int, dict[str, np.ndarray | float]]:
        """Parse multi-head outputs into per-horizon quantile returns and P(up)."""
        if self.multi_head_spec_ is None or self.primary_horizon_ is None:
            raise RuntimeError("predict_structured requires a multi-head model.")
        logits = self.predict_logits(x)
        parsed = parse_structured_predictions(
            logits,
            self.multi_head_spec_,
            primary_horizon=self.primary_horizon_,
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

    def save(self, path: str) -> None:
        """Persist weights, genome, scaler and optional calibration sidecars."""
        if self.genome_ is None or self.state_dict_ is None or self.scaler_ is None:
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

        path_obj.with_name(path_obj.name + GENOME_SUFFIX).write_text(
            json.dumps(self.genome_.to_dict(), separators=(",", ":")),
            encoding="utf-8",
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
    def load(cls, path: str, config: ModelConfig, task: str = "classification") -> EvolvedNNModel:
        """Load a saved evolved NN and its sidecars."""
        from sklearn.preprocessing import StandardScaler

        torch = _require_torch()
        model = cls(config, task=task)
        path_obj = Path(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model.state_dict_ = payload["state_dict"]
        model.best_iteration_ = payload.get("best_epoch")
        model.feature_names_ = list(payload.get("feature_names") or [])
        mh_payload = payload.get("multi_head")
        if mh_payload:
            model.multi_head_spec_ = MultiHeadSpec.from_dict(mh_payload)
            ph = payload.get("primary_horizon")
            model.primary_horizon_ = int(ph) if ph is not None else model.multi_head_spec_.horizons[-1]

        genome_path = path_obj.with_name(path_obj.name + GENOME_SUFFIX)
        if genome_path.exists():
            model.genome_ = NNGenome.from_dict(json.loads(genome_path.read_text(encoding="utf-8")))
        else:
            raise FileNotFoundError(f"Missing genome sidecar: {genome_path}")

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
            payload_cal = json.loads(sidecar.read_text(encoding="utf-8"))
            loaded = load_calibrator_sidecar(payload_cal)
            if isinstance(loaded, MultiHeadCalibrator):
                model.multi_calibrator_ = loaded
            else:
                model.calibrator_ = loaded
        return model

    def feature_importance(self) -> pd.Series:
        """Return cached permutation importances (empty when unavailable)."""
        if self._importance_cache is not None:
            return self._importance_cache
        if self.feature_names_ is None:
            raise RuntimeError("Model is not trained.")
        return pd.Series(0.0, index=self.feature_names_, name="permutation")


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "model.backend='evolved_nn' requires PyTorch. "
            "Install with `pip install torch` or `pip install -r requirements-optional.txt`."
        ) from exc
    return torch

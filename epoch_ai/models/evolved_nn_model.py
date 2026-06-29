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

from epoch_ai.config.settings import ModelConfig
from epoch_ai.models.base import BaseModel
from epoch_ai.models.calibration import ProbabilityCalibrator
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
    ) -> EvolvedNNModel:
        """Evolve architecture genes, train the best candidate, optionally calibrate."""
        if len(x) != len(y):
            raise ValueError("x and y must have the same length.")
        self.feature_names_ = list(x.columns)
        self.calibrator_ = None
        self._importance_cache = None
        self._infer_model = None
        self._infer_device = None

        if val_fraction is None:
            val_fraction = self.config.val_fraction

        x_arr = x.to_numpy(dtype=np.float64)
        y_arr = y.to_numpy(dtype=np.float64)
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

        cache = build_training_cache(
            x_arr,
            y_arr,
            self.config,
            task=self.task,
            sample_weight=sample_weight,
            val_fraction=val_fraction,
            split=split,
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

            for generation in range(evolution.generations):
                if use_parallel:
                    with ThreadPoolExecutor(max_workers=max_workers) as pool:
                        scores = list(pool.map(_train_candidate, population))
                else:
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

                # Human: bail out of evolution once it stops finding better architectures.
                # Agent: CONFIG evolution.early_stop_patience; null = run all generations.
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
            )
            if best_trained.val_loss < trained.val_loss:
                trained = best_trained

        self.genome_ = best_genome
        self.state_dict_ = trained.state_dict
        self.scaler_ = trained.scaler
        self.best_iteration_ = trained.best_epoch

        if self.task == "classification" and self.config.calibration != "none" and has_val:
            raw_val = predict_genome(
                x_arr[split:],
                self.genome_,
                self.state_dict_,
                self.scaler_,
                self.config,
                task=self.task,
            )
            self.calibrator_ = ProbabilityCalibrator.fit(
                raw_val,
                y_arr[split:],
                self.config.calibration,
            )

        if run_importance and has_val and self.task == "classification":
            imp = permutation_importance(
                x_arr[split:],
                y_arr[split:],
                self.genome_,
                self.state_dict_,
                self.scaler_,
                self.config,
                task=self.task,
                feature_names=self.feature_names_,
                rng=rng,
                model=self._inference_model(),
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
            self._infer_model = build_inference_model(
                input_dim,
                self.genome_,
                self.state_dict_,
                self.config,
                task=self.task,
                device=device,
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
        )
        if self.calibrator_ is not None:
            return self.calibrator_.transform(raw)
        return raw

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
        }
        torch.save(payload, path)

        path_obj.with_name(path_obj.name + GENOME_SUFFIX).write_text(
            json.dumps(self.genome_.to_dict(), indent=2),
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
                indent=2,
            ),
            encoding="utf-8",
        )

        sidecar = path_obj.with_name(path_obj.name + CALIBRATION_SUFFIX)
        if self.calibrator_ is not None:
            sidecar.write_text(
                json.dumps(self.calibrator_.to_dict(), indent=2),
                encoding="utf-8",
            )
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
            model.calibrator_ = ProbabilityCalibrator.from_dict(
                json.loads(sidecar.read_text(encoding="utf-8"))
            )
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

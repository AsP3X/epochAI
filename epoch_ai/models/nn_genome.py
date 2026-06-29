"""Neural-network genome encoding for evolutionary architecture search.

Each genome describes a candidate MLP topology and training hyper-parameters.
Evolution mutates these genes; gradient descent (Adam) fits the weights for each
candidate. Fitness is validation logloss on a time-ordered holdout tail.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from epoch_ai.config.settings import EvolutionConfig, NNConfig


def _clamp_layer_size(size: int, nn: NNConfig) -> int:
    """Clamp one hidden width to configured bounds."""
    return max(nn.hidden_size_min, min(nn.hidden_size_max, int(size)))


def _clamp_hidden_sizes(sizes: list[int], nn: NNConfig) -> list[int]:
    """Clamp each layer width and enforce min_layers / max_layers depth."""
    clamped = [_clamp_layer_size(s, nn) for s in sizes]
    if len(clamped) < nn.min_layers:
        clamped.extend([nn.hidden_size_min] * (nn.min_layers - len(clamped)))
    if len(clamped) > nn.max_layers:
        clamped = clamped[: nn.max_layers]
    return clamped


@dataclass(slots=True)
class NNGenome:
    """Architecture and optimizer hyper-parameters for one MLP candidate."""

    hidden_sizes: list[int]
    dropout: float
    learning_rate: float
    weight_decay: float
    use_batch_norm: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "hidden_sizes": list(self.hidden_sizes),
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "use_batch_norm": self.use_batch_norm,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> NNGenome:
        return cls(
            hidden_sizes=[int(x) for x in payload["hidden_sizes"]],  # type: ignore[arg-type]
            dropout=float(payload["dropout"]),
            learning_rate=float(payload["learning_rate"]),
            weight_decay=float(payload["weight_decay"]),
            use_batch_norm=bool(payload.get("use_batch_norm", True)),
        )


def default_genome(nn: NNConfig) -> NNGenome:
    """Sensible fixed architecture used when ``evolution.fast_fit`` is enabled."""
    if nn.fixed_hidden_sizes:
        hidden = _clamp_hidden_sizes(list(nn.fixed_hidden_sizes), nn)
    else:
        mid = max(nn.hidden_size_min, min(nn.hidden_size_max, 128))
        # Human: When min_layers > 2, default depth follows min_layers so fast_fit
        #        smokes reflect the configured depth floor.
        # Agent: READS nn.min_layers; default depth max(min_layers, 2) for backward compat.
        depth = max(nn.min_layers, 2)
        hidden: list[int] = []
        size = mid
        for _ in range(depth):
            hidden.append(_clamp_layer_size(size, nn))
            size = max(nn.hidden_size_min, size // 2)
    return NNGenome(
        hidden_sizes=hidden,
        dropout=0.15,
        learning_rate=1e-3,
        weight_decay=1e-4,
        use_batch_norm=True,
    )


def random_genome(rng: np.random.Generator, nn: NNConfig) -> NNGenome:
    """Sample a random valid genome within configured bounds."""
    if nn.fixed_hidden_sizes:
        hidden = _clamp_hidden_sizes(list(nn.fixed_hidden_sizes), nn)
    else:
        n_layers = int(rng.integers(nn.min_layers, nn.max_layers + 1))
        hidden = [
            int(rng.integers(nn.hidden_size_min, nn.hidden_size_max + 1)) for _ in range(n_layers)
        ]
        hidden = [_clamp_layer_size(h, nn) for h in hidden]
    log_lr = float(rng.uniform(math.log10(1e-4), math.log10(5e-3)))
    log_wd = float(rng.uniform(math.log10(1e-6), math.log10(1e-2)))
    return NNGenome(
        hidden_sizes=hidden,
        dropout=float(rng.uniform(0.0, 0.45)),
        learning_rate=10.0**log_lr,
        weight_decay=10.0**log_wd,
        use_batch_norm=bool(rng.integers(0, 2)),
    )


def mutate_genome(
    parent: NNGenome,
    rng: np.random.Generator,
    nn: NNConfig,
    *,
    sigma: float,
) -> NNGenome:
    """Perturb a parent genome to produce an offspring candidate."""
    child = NNGenome(
        hidden_sizes=list(parent.hidden_sizes),
        dropout=parent.dropout,
        learning_rate=parent.learning_rate,
        weight_decay=parent.weight_decay,
        use_batch_norm=parent.use_batch_norm,
    )

    roll = rng.random()
    if roll < 0.25 and len(child.hidden_sizes) < nn.max_layers:
        size = int(rng.integers(nn.hidden_size_min, nn.hidden_size_max + 1))
        child.hidden_sizes.append(size)
    elif roll < 0.40 and len(child.hidden_sizes) > nn.min_layers:
        child.hidden_sizes.pop(rng.integers(0, len(child.hidden_sizes)))
    elif roll < 0.55:
        idx = int(rng.integers(0, len(child.hidden_sizes)))
        delta = int(rng.normal(0, sigma * nn.hidden_size_max))
        child.hidden_sizes[idx] = _clamp_layer_size(child.hidden_sizes[idx] + delta, nn)
    elif roll < 0.70:
        child.dropout = float(np.clip(child.dropout + rng.normal(0, sigma * 0.2), 0.0, 0.5))
    elif roll < 0.85:
        child.learning_rate = float(
            np.clip(child.learning_rate * math.exp(rng.normal(0, sigma)), 1e-5, 1e-2)
        )
    elif roll < 0.95:
        child.weight_decay = float(
            np.clip(child.weight_decay * math.exp(rng.normal(0, sigma)), 1e-7, 1e-1)
        )
    else:
        child.use_batch_norm = not child.use_batch_norm

    return child


@dataclass
class EvolutionState:
    """Tracks the best genome discovered during a search."""

    population: list[NNGenome] = field(default_factory=list)
    fitness: list[float] = field(default_factory=list)
    best_genome: NNGenome | None = None
    best_fitness: float = float("inf")


def initialize_population(
    rng: np.random.Generator,
    nn: NNConfig,
    evolution: EvolutionConfig,
) -> list[NNGenome]:
    """Build the initial candidate pool (includes one default genome for stability)."""
    pop = [default_genome(nn)]
    while len(pop) < evolution.population_size:
        pop.append(random_genome(rng, nn))
    return pop


def initialize_population_from_seed(
    rng: np.random.Generator,
    nn: NNConfig,
    evolution: EvolutionConfig,
    seed: NNGenome,
) -> list[NNGenome]:
    """Warm-start evolution from a champion genome (elite + mutated offspring)."""
    pop = [seed]
    while len(pop) < evolution.population_size:
        pop.append(
            mutate_genome(
                seed,
                rng,
                nn,
                sigma=evolution.mutation_sigma,
            )
        )
    return pop

"""Tests for evolved_nn genome depth and width bounds."""

from __future__ import annotations

import numpy as np
import pytest

from epoch_ai.config.settings import NNConfig
from epoch_ai.models.nn_genome import default_genome, mutate_genome, random_genome


def test_default_genome_respects_min_layers():
    nn = NNConfig(min_layers=4, max_layers=6)
    genome = default_genome(nn)
    assert len(genome.hidden_sizes) == 4
    assert all(nn.hidden_size_min <= w <= nn.hidden_size_max for w in genome.hidden_sizes)


def test_default_genome_uses_fixed_hidden_sizes():
    nn = NNConfig(fixed_hidden_sizes=[256, 128, 64])
    genome = default_genome(nn)
    assert genome.hidden_sizes == [256, 128, 64]


def test_random_genome_depth_within_bounds():
    rng = np.random.default_rng(0)
    nn = NNConfig(min_layers=3, max_layers=5)
    for _ in range(20):
        genome = random_genome(rng, nn)
        assert nn.min_layers <= len(genome.hidden_sizes) <= nn.max_layers


def test_mutate_genome_does_not_shrink_below_min_layers():
    rng = np.random.default_rng(1)
    nn = NNConfig(min_layers=3, max_layers=6)
    parent = default_genome(nn)
    for _ in range(50):
        child = mutate_genome(parent, rng, nn, sigma=0.5)
        assert len(child.hidden_sizes) >= nn.min_layers
        assert len(child.hidden_sizes) <= nn.max_layers


def test_nn_config_rejects_min_layers_above_max():
    with pytest.raises(ValueError, match="min_layers"):
        NNConfig(min_layers=5, max_layers=3)


def test_nn_config_rejects_fixed_sizes_outside_depth():
    with pytest.raises(ValueError, match="fixed_hidden_sizes"):
        NNConfig(min_layers=2, max_layers=3, fixed_hidden_sizes=[64, 128, 256, 512])

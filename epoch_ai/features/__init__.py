"""Modular feature-engineering system."""

from __future__ import annotations

from epoch_ai.features.base import FeatureGroup, build_feature_groups
from epoch_ai.features.pipeline import FeaturePipeline, build_target

__all__ = [
    "FeatureGroup",
    "FeaturePipeline",
    "build_feature_groups",
    "build_target",
]

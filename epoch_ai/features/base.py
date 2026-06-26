"""Feature-group abstraction and registry.

Each :class:`FeatureGroup` transforms a raw OHLCV(+context) frame into a set of
engineered columns. Groups are intentionally small and self-contained so new feature
families can be added without touching the rest of the system. The
:func:`build_feature_groups` factory assembles the active groups from config toggles.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import pandas as pd

from epoch_ai.config.settings import FeatureConfig


class FeatureGroup(ABC):
    """Base class for a named group of engineered features."""

    #: Short, unique name used for logging and prefixes.
    name: str = "base"

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute this group's features.

        Args:
            df: Raw OHLCV(+context) frame indexed by ``timestamp``.

        Returns:
            A DataFrame of engineered columns sharing ``df``'s index. Implementations
            must be **causal** (use only information available up to each row) so the
            features are valid for walk-forward prediction.
        """
        raise NotImplementedError


def build_feature_groups(
    config: FeatureConfig,
    *,
    context_symbols: Sequence[str] | None = None,
) -> list[FeatureGroup]:
    """Instantiate the feature groups enabled in ``config``.

    Indicator look-back windows are threaded through from ``config`` so the feature
    set is fully config-driven. Imports are local to avoid importing every group when
    only a subset is enabled.
    """
    from epoch_ai.features.cross_asset import CrossAssetFeatures
    from epoch_ai.features.derivatives import DerivativesFeatures
    from epoch_ai.features.microstructure import MicrostructureFeatures
    from epoch_ai.features.onchain import OnChainFeatures
    from epoch_ai.features.sentiment import SentimentFeatures
    from epoch_ai.features.technical import TechnicalFeatures
    from epoch_ai.features.time_features import TimeFeatures
    from epoch_ai.features.volatility import VolatilityFeatures

    groups: list[FeatureGroup] = []
    if config.technical:
        groups.append(
            TechnicalFeatures(
                return_lags=config.return_lags,
                ma_windows=config.ma_windows,
                rsi_periods=config.rsi_periods,
            )
        )
    if config.microstructure:
        groups.append(MicrostructureFeatures())
    if config.derivatives:
        groups.append(DerivativesFeatures())
    if config.volatility:
        groups.append(VolatilityFeatures(vol_windows=config.vol_windows))
    if config.time:
        groups.append(TimeFeatures())
    if config.sentiment:
        groups.append(SentimentFeatures())
    if config.onchain:
        groups.append(OnChainFeatures())
    if config.cross_asset:
        ctx = context_symbols or ()
        groups.append(
            CrossAssetFeatures(
                context_symbols=ctx,
                return_lags=config.return_lags,
            )
        )
    if config.patterns:
        from epoch_ai.features.patterns import PatternFeatures

        groups.append(
            PatternFeatures(
                lookbacks=config.pattern_lookbacks,
                pivot_confirm_bars=config.pivot_confirm_bars,
            )
        )
    if config.manipulation:
        from epoch_ai.features.manipulation import ManipulationFeatures

        groups.append(ManipulationFeatures())
    return groups

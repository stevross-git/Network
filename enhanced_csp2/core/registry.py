from __future__ import annotations

from typing import Dict, Set

from .interfaces import FeatureProvider


class FeatureRegistry:
    """Registry for optional features."""

    def __init__(self) -> None:
        self._features: Dict[str, FeatureProvider] = {}
        self._enabled: Set[str] = set()

    def register_feature(self, feature: FeatureProvider, enabled: bool = False) -> None:
        """Register a feature provider."""
        self._features[feature.name] = feature
        if enabled:
            self._enabled.add(feature.name)

    async def start_enabled_features(self) -> None:
        """Initialize all enabled features."""
        for name in list(self._enabled):
            feature = self._features.get(name)
            if feature:
                await feature.initialize()

    async def shutdown(self) -> None:
        """Shutdown all enabled features."""
        for name in list(self._enabled):
            feature = self._features.get(name)
            if feature:
                await feature.shutdown()

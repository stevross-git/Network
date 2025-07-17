from __future__ import annotations

from typing import Optional

from .core.minimal_node import MinimalNetworkNode
from ..core.config import NetworkConfig
from ..core.registry import FeatureRegistry


class EnhancedNetworkNode(MinimalNetworkNode):
    """Network node with optional features."""

    def __init__(self, config: NetworkConfig) -> None:
        super().__init__(config.core)
        self.feature_config = config.features
        self.feature_registry = FeatureRegistry()

    async def start(self) -> None:
        await super().start()
        await self.feature_registry.start_enabled_features()

    async def stop(self) -> None:
        await self.feature_registry.shutdown()
        await super().stop()

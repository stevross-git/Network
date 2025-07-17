from __future__ import annotations

import logging
from typing import Set, Type

from ..network.enhanced_node import EnhancedNetworkNode
from ..core.config import FeatureConfig
from ..core.registry import FeatureRegistry
from ..core.interfaces import FeatureProvider

logger = logging.getLogger(__name__)


class ProgressiveFeatureLoader:
    """Load features progressively based on configuration."""

    def __init__(self, network_node: EnhancedNetworkNode) -> None:
        self.network = network_node
        self.loaded_features: Set[str] = set()

    async def _try_load_feature(self, name: str, cls: Type[FeatureProvider]) -> bool:
        try:
            feature = cls(self.network)
            await feature.initialize()
            self.network.feature_registry.register_feature(feature, enabled=True)
            self.loaded_features.add(name)
            logger.info("Loaded feature: %s", name)
            return True
        except Exception as exc:
            logger.warning("Feature %s not available: %s", name, exc)
            return False

    async def load_features(self, config: FeatureConfig) -> None:
        if config.enable_compression:
            from ..features.compression import CompressionFeature
            await self._try_load_feature("compression", CompressionFeature)
        if config.enable_batching:
            from ..features.batching import BatchingFeature
            await self._try_load_feature("batching", BatchingFeature)
        if config.enable_encryption:
            from ..features.encryption import EncryptionFeature
            await self._try_load_feature("encryption", EncryptionFeature)
        if config.enable_advanced_security:
            from ..features.security import SecurityFeature
            await self._try_load_feature("security", SecurityFeature)
        if config.enable_ai:
            from ..features.ai.ai_feature import AIFeature
            await self._try_load_feature("ai", AIFeature)
        if config.enable_quantum:
            from ..features.quantum.quantum_feature import QuantumFeature
            await self._try_load_feature("quantum", QuantumFeature)
        if config.enable_blockchain:
            from ..features.blockchain.blockchain_feature import BlockchainFeature
            await self._try_load_feature("blockchain", BlockchainFeature)

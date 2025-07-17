"""Standard CSP network with basic features."""

import asyncio
import logging

from ..core.config import NetworkConfig
from ..network.enhanced_node import EnhancedNetworkNode
from .feature_loader import ProgressiveFeatureLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main() -> None:
    config = NetworkConfig.standard()
    node = EnhancedNetworkNode(config)
    loader = ProgressiveFeatureLoader(node)
    try:
        await node.start()
        await loader.load_features(config.features)
        logger.info("Standard CSP network started with features: %s", loader.loaded_features)
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await node.stop()


if __name__ == "__main__":
    asyncio.run(main())

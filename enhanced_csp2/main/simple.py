"""Minimal CSP network entry point."""

import asyncio
import logging

from ..core.config import CoreConfig
from ..network.core.minimal_node import MinimalNetworkNode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main() -> None:
    config = CoreConfig()
    node = MinimalNetworkNode(config)
    try:
        await node.start()
        logger.info("Minimal CSP network started")
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await node.stop()


if __name__ == "__main__":
    asyncio.run(main())

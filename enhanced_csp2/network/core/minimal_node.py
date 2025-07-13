from __future__ import annotations

from typing import Dict, Optional

from ...core.config import CoreConfig
from ...core.types import NodeID, PeerInfo
from ...core.interfaces import NetworkTransport


class MinimalNetworkNode:
    """Minimal peer-to-peer network node."""

    def __init__(self, config: CoreConfig) -> None:
        self.config = config
        self.node_id = NodeID.generate()
        self.peers: Dict[str, PeerInfo] = {}
        self.is_running = False
        self.transport: Optional[NetworkTransport] = None

    async def start(self) -> None:
        if self.transport:
            await self.transport.start()
        self.is_running = True

    async def stop(self) -> None:
        self.is_running = False
        if self.transport:
            await self.transport.stop()

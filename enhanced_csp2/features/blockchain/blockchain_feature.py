from __future__ import annotations

from ..core.interfaces import FeatureProvider


class BlockchainFeature(FeatureProvider):
    @property
    def name(self) -> str:
        return "blockchain"

    def __init__(self, node) -> None:
        self.node = node
        self.client = None

    async def initialize(self) -> None:
        try:
            from web3 import Web3  # type: ignore
            self.client = object()
        except Exception as exc:
            raise RuntimeError("Blockchain dependencies not installed") from exc

    async def shutdown(self) -> None:
        self.client = None

from __future__ import annotations

from ..core.interfaces import FeatureProvider


class SecurityFeature(FeatureProvider):
    @property
    def name(self) -> str:
        return "security"

    def __init__(self, node) -> None:
        self.node = node

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

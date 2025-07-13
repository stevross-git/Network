from __future__ import annotations

from ..core.interfaces import FeatureProvider


class AIFeature(FeatureProvider):
    @property
    def name(self) -> str:
        return "ai"

    def __init__(self, node) -> None:
        self.node = node
        self.engine = None

    async def initialize(self) -> None:
        try:
            from transformers import AutoModel  # type: ignore
            self.engine = object()
        except Exception as exc:
            raise RuntimeError("AI dependencies not installed") from exc

    async def shutdown(self) -> None:
        self.engine = None

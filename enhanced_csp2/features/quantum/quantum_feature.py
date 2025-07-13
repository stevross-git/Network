from __future__ import annotations

from ..core.interfaces import FeatureProvider


class QuantumFeature(FeatureProvider):
    @property
    def name(self) -> str:
        return "quantum"

    def __init__(self, node) -> None:
        self.node = node
        self.engine = None

    async def initialize(self) -> None:
        try:
            from qiskit import QuantumCircuit  # type: ignore
            self.engine = object()
        except Exception as exc:
            raise RuntimeError("Quantum dependencies not installed") from exc

    async def shutdown(self) -> None:
        self.engine = None

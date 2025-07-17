from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Protocol


class NetworkTransport(Protocol):
    """Protocol for network transport implementations."""

    async def start(self) -> None:
        ...

    async def stop(self) -> None:
        ...

    async def send_message(self, peer_id: str, message: bytes) -> None:
        ...


class SecurityProvider(Protocol):
    """Protocol for security providers."""

    async def validate_message(self, message: bytes) -> bool:
        ...

    async def encrypt_message(self, message: bytes) -> bytes:
        ...


class FeatureProvider(Protocol):
    """Protocol for optional feature modules."""

    @property
    def name(self) -> str:
        ...

    async def initialize(self) -> None:
        ...

    async def shutdown(self) -> None:
        ...

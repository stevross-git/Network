from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class MessageType(str, Enum):
    DATA = "data"
    CONTROL = "control"
    HEARTBEAT = "heartbeat"


@dataclass
class NodeID:
    value: str

    @classmethod
    def generate(cls) -> "NodeID":
        return cls(value=str(uuid.uuid4()))

    def __str__(self) -> str:
        return self.value


@dataclass
class PeerInfo:
    id: NodeID
    address: str
    port: int
    metadata: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if isinstance(self.id, str):
            self.id = NodeID(self.id)
        if self.metadata is None:
            self.metadata = {}


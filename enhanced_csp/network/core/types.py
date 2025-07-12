# enhanced_csp/network/core/types.py
"""Core type definitions for Enhanced CSP Network."""

from __future__ import annotations
import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, List, Union
import base58

# Import cryptography safely
try:
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.hazmat.primitives import serialization
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    ed25519 = None
    serialization = None

# Import secure_random utility safely
try:
    from ..utils.secure_random import secure_bytes
except ImportError:
    import secrets
    def secure_bytes(length: int) -> bytes:
        return secrets.token_bytes(length)


class MessageType(Enum):
    """Types of network messages."""
    DATA = "data"
    CONTROL = "control"
    HEARTBEAT = "heartbeat"
    DISCOVERY = "discovery"
    ROUTING = "routing"
    AUTH = "auth"
    ERROR = "error"


@dataclass
class NodeID:
    """Unique identifier for network nodes."""
    
    value: str = ""
    raw_id: Optional[bytes] = None
    public_key: Optional[Any] = None  # ed25519.Ed25519PublicKey if available
    
    def __post_init__(self):
        if not self.value and self.raw_id:
            # Convert raw_id to base58 string
            self.value = base58.b58encode(self.raw_id).decode('ascii')
        elif not self.value:
            # Generate a random node ID if none provided
            self.value = self.generate().value
    
    @classmethod
    def generate(cls) -> "NodeID":
        """Generate a new random NodeID."""
        # Create a random 32-byte hash with multihash prefix
        random_bytes = secure_bytes(32)
        raw = b"\x12\x20" + hashlib.sha256(random_bytes).digest()
        value = base58.b58encode(raw).decode('ascii')
        return cls(value=value, raw_id=raw)

    @classmethod
    def from_string(cls, value: str) -> "NodeID":
        """Create NodeID from string."""
        try:
            raw_id = base58.b58decode(value)
            return cls(value=value, raw_id=raw_id)
        except Exception:
            # If base58 decode fails, just use the string as-is
            return cls(value=value)

    @classmethod
    def from_public_key(cls, public_key) -> "NodeID":
        """Create NodeID from an Ed25519 public key."""
        if not CRYPTOGRAPHY_AVAILABLE or not public_key:
            return cls.generate()
        
        try:
            pk_bytes = public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
            raw = b"\x12\x20" + hashlib.sha256(pk_bytes).digest()
            value = base58.b58encode(raw).decode('ascii')
            return cls(value=value, raw_id=raw, public_key=public_key)
        except Exception:
            return cls.generate()

    def to_base58(self) -> str:
        """Return the base58 representation."""
        return self.value
    
    def __str__(self) -> str:
        return self.value
    
    def __repr__(self) -> str:
        return f"NodeID('{self.value}')"
    
    def __eq__(self, other) -> bool:
        if isinstance(other, NodeID):
            return self.value == other.value
        elif isinstance(other, str):
            return self.value == other
        return False
    
    def __hash__(self) -> int:
        return hash(self.value)


@dataclass
class NodeCapabilities:
    """Node capability flags."""
    relay: bool = False
    storage: bool = False
    compute: bool = False
    quantum: bool = False
    blockchain: bool = False
    dns: bool = False
    bootstrap: bool = False
    ai: bool = False
    mesh_routing: bool = False
    nat_traversal: bool = False
    
    def to_dict(self) -> Dict[str, bool]:
        """Convert to dictionary."""
        return {
            'relay': self.relay,
            'storage': self.storage,
            'compute': self.compute,
            'quantum': self.quantum,
            'blockchain': self.blockchain,
            'dns': self.dns,
            'bootstrap': self.bootstrap,
            'ai': self.ai,
            'mesh_routing': self.mesh_routing,
            'nat_traversal': self.nat_traversal,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, bool]) -> "NodeCapabilities":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class PeerInfo:
    """Information about a peer node."""
    id: NodeID
    address: str
    port: int
    capabilities: NodeCapabilities = field(default_factory=NodeCapabilities)
    last_seen: datetime = field(default_factory=datetime.utcnow)
    latency: Optional[float] = None
    reputation: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    protocols: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    
    def __post_init__(self):
        # Ensure id is a NodeID instance
        if isinstance(self.id, str):
            self.id = NodeID.from_string(self.id)
        
        # Ensure capabilities is a NodeCapabilities instance
        if isinstance(self.capabilities, dict):
            self.capabilities = NodeCapabilities.from_dict(self.capabilities)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': str(self.id),
            'address': self.address,
            'port': self.port,
            'capabilities': self.capabilities.to_dict(),
            'last_seen': self.last_seen.isoformat(),
            'latency': self.latency,
            'reputation': self.reputation,
            'metadata': self.metadata,
            'protocols': self.protocols,
            'version': self.version,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PeerInfo":
        """Create from dictionary."""
        # Parse last_seen
        last_seen = data.get('last_seen')
        if isinstance(last_seen, str):
            try:
                last_seen = datetime.fromisoformat(last_seen)
            except ValueError:
                last_seen = datetime.utcnow()
        elif last_seen is None:
            last_seen = datetime.utcnow()
        
        return cls(
            id=NodeID.from_string(data['id']),
            address=data['address'],
            port=data['port'],
            capabilities=NodeCapabilities.from_dict(data.get('capabilities', {})),
            last_seen=last_seen,
            latency=data.get('latency'),
            reputation=data.get('reputation', 1.0),
            metadata=data.get('metadata', {}),
            protocols=data.get('protocols', []),
            version=data.get('version', '1.0.0'),
        )


@dataclass
class NetworkMessage:
    """Network message structure."""
    type: MessageType
    sender: NodeID
    recipient: Optional[NodeID] = None
    payload: Any = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    ttl: int = 64
    signature: Optional[bytes] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: int = 0
    compressed: bool = False
    encrypted: bool = False
    
    def __post_init__(self):
        # Ensure sender is a NodeID instance
        if isinstance(self.sender, str):
            self.sender = NodeID.from_string(self.sender)
        
        # Ensure recipient is a NodeID instance if provided
        if self.recipient and isinstance(self.recipient, str):
            self.recipient = NodeID.from_string(self.recipient)
        
        # Ensure type is MessageType enum
        if isinstance(self.type, str):
            try:
                self.type = MessageType(self.type)
            except ValueError:
                self.type = MessageType.DATA
    
    @classmethod
    def create(cls, msg_type: MessageType, sender: Union[NodeID, str], payload: Any,
               recipient: Optional[Union[NodeID, str]] = None, ttl: int = 64,
               priority: int = 0) -> 'NetworkMessage':
        """Create a new network message."""
        # Convert string inputs to proper types
        if isinstance(sender, str):
            sender = NodeID.from_string(sender)
        if isinstance(recipient, str):
            recipient = NodeID.from_string(recipient)
        
        return cls(
            id=str(uuid.uuid4()),
            type=msg_type,
            sender=sender,
            recipient=recipient,
            payload=payload,
            timestamp=datetime.utcnow(),
            ttl=ttl,
            priority=priority,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'type': self.type.value,
            'sender': str(self.sender),
            'recipient': str(self.recipient) if self.recipient else None,
            'payload': self.payload,
            'timestamp': self.timestamp.isoformat(),
            'ttl': self.ttl,
            'signature': self.signature.hex() if self.signature else None,
            'priority': self.priority,
            'compressed': self.compressed,
            'encrypted': self.encrypted,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NetworkMessage":
        """Create from dictionary."""
        # Parse timestamp
        timestamp = data.get('timestamp')
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp)
            except ValueError:
                timestamp = datetime.utcnow()
        elif timestamp is None:
            timestamp = datetime.utcnow()
        
        # Parse signature
        signature = data.get('signature')
        if isinstance(signature, str):
            try:
                signature = bytes.fromhex(signature)
            except ValueError:
                signature = None
        
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            type=MessageType(data['type']),
            sender=NodeID.from_string(data['sender']),
            recipient=NodeID.from_string(data['recipient']) if data.get('recipient') else None,
            payload=data.get('payload'),
            timestamp=timestamp,
            ttl=data.get('ttl', 64),
            signature=signature,
            priority=data.get('priority', 0),
            compressed=data.get('compressed', False),
            encrypted=data.get('encrypted', False),
        )
    
    def is_expired(self) -> bool:
        """Check if message has expired based on TTL."""
        if self.ttl <= 0:
            return True
        
        # Calculate age in seconds
        age = (datetime.utcnow() - self.timestamp).total_seconds()
        return age > self.ttl * 10  # TTL is in hops, estimate 10 seconds per hop
    
    def decrement_ttl(self) -> None:
        """Decrement TTL when forwarding message."""
        self.ttl = max(0, self.ttl - 1)


# Connection and transport types
@dataclass
class ConnectionInfo:
    """Information about a network connection."""
    peer_id: NodeID
    local_address: str
    remote_address: str
    protocol: str
    established_at: datetime = field(default_factory=datetime.utcnow)
    bytes_sent: int = 0
    bytes_received: int = 0
    last_activity: datetime = field(default_factory=datetime.utcnow)
    status: str = "active"
    
    def update_activity(self, bytes_sent: int = 0, bytes_received: int = 0):
        """Update connection activity."""
        self.bytes_sent += bytes_sent
        self.bytes_received += bytes_received
        self.last_activity = datetime.utcnow()


# Error and status types
class NetworkStatus(Enum):
    """Network node status."""
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class NetworkError:
    """Network error information."""
    code: str
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: Optional[NodeID] = None
    details: Dict[str, Any] = field(default_factory=dict)


# Route and topology types
@dataclass
class RouteInfo:
    """Information about a network route."""
    destination: NodeID
    next_hop: NodeID
    metric: float
    hops: int
    last_updated: datetime = field(default_factory=datetime.utcnow)
    protocol: str = "batman"
    
    def is_stale(self, max_age_seconds: int = 300) -> bool:
        """Check if route information is stale."""
        age = (datetime.utcnow() - self.last_updated).total_seconds()
        return age > max_age_seconds


# Export all types
__all__ = [
    "MessageType",
    "NodeID",
    "NodeCapabilities",
    "PeerInfo",
    "NetworkMessage",
    "ConnectionInfo",
    "NetworkStatus",
    "NetworkError",
    "RouteInfo",
]
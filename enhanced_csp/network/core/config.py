# enhanced_csp/network/core/config.py
"""Network configuration dataclasses for Enhanced CSP."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from pathlib import Path
import json

# Proper imports instead of dynamic imports
if TYPE_CHECKING:
    from .types import NodeCapabilities
else:
    try:
        from .types import NodeCapabilities
    except ImportError:
        # Create a fallback class for development
        class NodeCapabilities:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

try:
    import yaml
except ImportError:
    yaml = None

NIST_KEM_ALGORITHMS = {"ML-KEM-768"}
NIST_SIGNATURE_ALGORITHMS = {"ML-DSA-65"}
NIST_BACKUP_SIGNATURES = {"SLH-DSA-SHAKE-128s"}


def _normalize_algorithm(value: str, mapping: Dict[str, str]) -> str:
    value_lower = value.lower().strip()
    return mapping.get(value_lower, value)


@dataclass
class PQCConfig:
    """Post-quantum cryptography settings."""

    kem_algorithm: str = "ML-KEM-768"
    """Key encapsulation mechanism algorithm."""

    signature_algorithm: str = "ML-DSA-65"
    """Digital signature algorithm."""

    backup_signature: str = "SLH-DSA-SHAKE-128s"
    """Backup signature scheme used as fallback."""

    key_rotation_interval: int = 86400 * 7
    """Interval in seconds to rotate PQC keys."""

    enable_hybrid_mode: bool = True
    """Use classical algorithms in parallel with PQC."""

    def __post_init__(self) -> None:
        old_kem = {"kyber768": "ML-KEM-768"}
        old_sig = {"dilithium3": "ML-DSA-65"}
        old_backup = {"sphincs+": "SLH-DSA-SHAKE-128s"}

        self.kem_algorithm = _normalize_algorithm(self.kem_algorithm, old_kem)
        self.signature_algorithm = _normalize_algorithm(
            self.signature_algorithm, old_sig
        )
        self.backup_signature = _normalize_algorithm(self.backup_signature, old_backup)

        if self.kem_algorithm not in NIST_KEM_ALGORITHMS:
            raise ValueError(f"Unsupported KEM algorithm: {self.kem_algorithm}")
        if self.signature_algorithm not in NIST_SIGNATURE_ALGORITHMS:
            raise ValueError(
                f"Unsupported signature algorithm: {self.signature_algorithm}"
            )
        if self.backup_signature not in NIST_BACKUP_SIGNATURES:
            raise ValueError(
                f"Unsupported backup signature: {self.backup_signature}"
            )
        if self.key_rotation_interval <= 0:
            raise ValueError("key_rotation_interval must be positive")


@dataclass
class SecurityConfig:
    """Security related settings."""

    enable_tls: bool = True
    """Enable TLS for all transports."""

    enable_mtls: bool = False
    """Require mutual TLS authentication."""

    tls_version: str = "1.3"
    """TLS protocol version."""

    tls_cert_path: Optional[Path] = None
    """Path to the TLS certificate file."""

    tls_key_path: Optional[Path] = None
    """Path to the TLS private key file."""

    ca_cert_path: Optional[Path] = None
    """Path to the CA certificate file."""

    enable_pq_crypto: bool = True
    """Enable post-quantum cryptography features."""

    pqc: PQCConfig = field(default_factory=PQCConfig)
    """Post-quantum cryptography configuration."""

    def __post_init__(self) -> None:
        for attr in ("tls_cert_path", "tls_key_path", "ca_cert_path"):
            value = getattr(self, attr)
            if isinstance(value, str):
                value = Path(value)
                object.__setattr__(self, attr, value)
            if value is not None and not value.exists():
                raise ValueError(f"{attr} does not exist: {value}")
        if isinstance(self.pqc, dict):
            object.__setattr__(self, "pqc", PQCConfig(**self.pqc))
        self.pqc.__post_init__()


@dataclass
class P2PConfig:
    """Peer-to-peer connectivity settings."""

    listen_address: str = "0.0.0.0"
    """IP address to bind to."""

    listen_port: int = 9000
    """Port for incoming connections."""

    enable_quic: bool = True
    """Enable QUIC transport."""

    enable_tcp: bool = True
    enable_websocket: bool = False
    enable_mdns: bool = True

    bootstrap_nodes: List[str] = field(default_factory=list)
    bootstrap_api_url: Optional[str] = None
    dns_seed_domain: Optional[str] = None
    stun_servers: List[str] = field(
        default_factory=lambda: [
            "stun:stun.l.google.com:19302",
            "stun:global.stun.twilio.com:3478",
        ]
    )
    turn_servers: List[Dict[str, Any]] = field(default_factory=list)

    connection_timeout: int = 30
    max_connections: int = 100
    min_peers: int = 3
    max_peers: int = 50
    max_message_size: int = 1024 * 1024
    """Maximum size of a message in bytes."""

    def __post_init__(self) -> None:
        from ..utils import validate_ip_address, validate_port_number

        validate_ip_address(self.listen_address)
        self.listen_port = validate_port_number(self.listen_port)
        if self.max_message_size <= 0:
            raise ValueError("max_message_size must be positive")


@dataclass
class MeshConfig:
    """Mesh topology parameters."""

    topology_type: str = "dynamic_partial"
    """Type of mesh topology to use."""

    enable_super_peers: bool = True
    super_peer_capacity_threshold: float = 100.0
    max_peers: int = 20
    routing_update_interval: int = 10
    link_quality_threshold: float = 0.5
    enable_multi_hop: bool = True
    max_hop_count: int = 10

    def __post_init__(self) -> None:
        if self.max_peers <= 0:
            raise ValueError("max_peers must be positive")


@dataclass
class DNSConfig:
    """Distributed DNS overlay settings."""

    root_domain: str = ".web4ai"
    """Root domain used for DNS entries."""

    enable_dnssec: bool = True
    default_ttl: int = 3600
    cache_size: int = 10000
    enable_recursive: bool = True
    upstream_dns: List[str] = field(default_factory=lambda: ["8.8.8.8", "1.1.1.1"])

    def __post_init__(self) -> None:
        if not self.root_domain.startswith('.'):
            raise ValueError("root_domain must start with '.'")


@dataclass
class RoutingConfig:
    """Settings for adaptive routing."""

    enable_multipath: bool = True
    enable_ml_predictor: bool = True
    max_paths_per_destination: int = 3
    failover_threshold_ms: int = 500
    path_quality_update_interval: int = 30
    metric_update_interval: int = 30
    route_optimization_interval: int = 60
    ml_update_interval: int = 300  # ML training interval in seconds
    enable_congestion_control: bool = True
    enable_qos: bool = True
    priority_levels: int = 4

    def __post_init__(self) -> None:
        if self.max_paths_per_destination <= 0:
            raise ValueError("max_paths_per_destination must be positive")


@dataclass
class NetworkConfig:
    """Main network configuration."""
    
    # Your existing fields...
    security: SecurityConfig = field(default_factory=lambda: SecurityConfig())
    p2p: P2PConfig = field(default_factory=lambda: P2PConfig())
    mesh: MeshConfig = field(default_factory=lambda: MeshConfig())
    dns: DNSConfig = field(default_factory=lambda: DNSConfig())
    routing: RoutingConfig = field(default_factory=lambda: RoutingConfig())
    
    node_name: str = "csp-node"
    node_type: str = "standard"
    capabilities: NodeCapabilities = field(default_factory=lambda: NodeCapabilities())
    data_dir: Path = field(default_factory=lambda: Path("./data"))
    
    # 🚨 ADD THESE MISSING FIELDS:
    network_id: str = "enhanced-csp-network"
    listen_address: str = "0.0.0.0"
    listen_port: int = 9000
    
    # Core feature enablement flags
    enable_discovery: bool = True
    enable_dht: bool = True
    enable_nat_traversal: bool = True
    enable_mesh: bool = True
    enable_dns: bool = True
    enable_adaptive_routing: bool = True
    enable_routing: bool = True
    enable_metrics: bool = True
    enable_compression: bool = True
    
    # Advanced features
    enable_storage: bool = True
    enable_quantum: bool = True
    enable_blockchain: bool = False
    enable_compute: bool = True
    enable_ai: bool = True
    
    def __post_init__(self) -> None:
        """Validate and initialize configuration."""
        # Your existing validation code...
        if not self.node_name or not isinstance(self.node_name, str):
            raise ValueError("node_name must be a non-empty string")
        
        self.data_dir = Path(self.data_dir).expanduser()
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Sync listen_port between main config and p2p config
        if hasattr(self.p2p, 'listen_port') and self.p2p.listen_port != self.listen_port:
            self.p2p.listen_port = self.listen_port
        
        # Initialize sub-configs
        for config in [self.p2p, self.mesh, self.dns, self.routing]:
            if hasattr(config, '__post_init__'):
                config.__post_init__()

    # ------------------------------------------------------------------
    # Enhanced factory methods
    # ------------------------------------------------------------------
    @classmethod
    def development(cls) -> 'NetworkConfig':
        """Configuration suitable for development."""
        return cls(
            node_name="dev-node",
            data_dir=Path("./dev_data"),
            listen_port=9000,
            # Enable all features for development
            enable_discovery=True,
            enable_dht=True,
            enable_mesh=True,
            enable_dns=True,
            enable_adaptive_routing=True,
            enable_storage=True,
            enable_compute=True,
        )

    @classmethod
    def production(cls) -> 'NetworkConfig':
        """Configuration with production defaults."""
        return cls(
            node_name="prod-node",
            listen_port=30301,
            # Full production features
            enable_discovery=True,
            enable_dht=True,
            enable_mesh=True,
            enable_dns=True,
            enable_adaptive_routing=True,
            enable_storage=True,
            enable_compute=True,
            enable_quantum=True,
        )

    @classmethod
    def test(cls) -> 'NetworkConfig':
        """Configuration for testing environments."""
        return cls(
            node_name="test-node",
            data_dir=Path("./test_data"),
            listen_port=0,  # Random port for testing
            # Minimal features for testing
            enable_discovery=True,
            enable_dht=False,
            enable_mesh=True,
            enable_dns=False,
        )
    
    @classmethod
    def genesis_node(cls) -> 'NetworkConfig':
        """Configuration for a genesis/bootstrap node."""
        return cls(
            node_name="genesis-node",
            node_type="genesis",
            listen_port=30300,
            capabilities=NodeCapabilities(
                relay=True,
                storage=True,
                compute=True,
                quantum=True,
                dns=True,
                bootstrap=True,
                ai=True,
                mesh_routing=True,
                nat_traversal=True,
            ),
            # Genesis nodes need all features
            enable_discovery=True,
            enable_dht=True,
            enable_mesh=True,
            enable_dns=True,
            enable_adaptive_routing=True,
            enable_storage=True,
            enable_compute=True,
        )

    # ------------------------------------------------------------------
    # (De)serialisation helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NetworkConfig':
        return cls(
            security=SecurityConfig(**data.get('security', {})),
            p2p=P2PConfig(**data.get('p2p', {})),
            mesh=MeshConfig(**data.get('mesh', {})),
            dns=DNSConfig(**data.get('dns', {})),
            routing=RoutingConfig(**data.get('routing', {})),
            node_name=data.get('node_name', 'csp-node'),
            node_type=data.get('node_type', 'standard'),
            capabilities=NodeCapabilities(**data.get('capabilities', {})),
            data_dir=Path(data.get('data_dir', './data')),
            # Feature flags
            network_id=data.get('network_id', 'enhanced-csp-network'),
            listen_address=data.get('listen_address', '0.0.0.0'),
            listen_port=data.get('listen_port', 9000),
            enable_discovery=data.get('enable_discovery', True),
            enable_dht=data.get('enable_dht', True),
            enable_nat_traversal=data.get('enable_nat_traversal', True),
            enable_mesh=data.get('enable_mesh', True),
            enable_dns=data.get('enable_dns', True),
            enable_adaptive_routing=data.get('enable_adaptive_routing', True),
            enable_routing=data.get('enable_routing', True),
            enable_metrics=data.get('enable_metrics', True),
            enable_compression=data.get('enable_compression', True),
            enable_storage=data.get('enable_storage', True),
            enable_quantum=data.get('enable_quantum', True),
            enable_blockchain=data.get('enable_blockchain', False),
            enable_compute=data.get('enable_compute', True),
            enable_ai=data.get('enable_ai', True),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'security': asdict(self.security),
            'p2p': asdict(self.p2p),
            'mesh': asdict(self.mesh),
            'dns': asdict(self.dns),
            'routing': asdict(self.routing),
            'node_name': self.node_name,
            'node_type': self.node_type,
            'capabilities': asdict(self.capabilities),
            'data_dir': str(self.data_dir),
            'network_id': self.network_id,
            'listen_address': self.listen_address,
            'listen_port': self.listen_port,
            'enable_discovery': self.enable_discovery,
            'enable_dht': self.enable_dht,
            'enable_nat_traversal': self.enable_nat_traversal,
            'enable_mesh': self.enable_mesh,
            'enable_dns': self.enable_dns,
            'enable_adaptive_routing': self.enable_adaptive_routing,
            'enable_routing': self.enable_routing,
            'enable_metrics': self.enable_metrics,
            'enable_compression': self.enable_compression,
            'enable_storage': self.enable_storage,
            'enable_quantum': self.enable_quantum,
            'enable_blockchain': self.enable_blockchain,
            'enable_compute': self.enable_compute,
            'enable_ai': self.enable_ai,
        }

    @classmethod
    def load(cls, path: Path) -> 'NetworkConfig':
        path = Path(path)
        with path.open('r', encoding='utf-8') as fh:
            if path.suffix.lower() in {'.json'} or yaml is None:
                data = json.load(fh)
            else:
                data = yaml.safe_load(fh)
        return cls.from_dict(data)

    def save(self, path: Path) -> None:
        path = Path(path)
        with path.open('w', encoding='utf-8') as fh:
            if path.suffix.lower() in {'.json'} or yaml is None:
                json.dump(self.to_dict(), fh, indent=2)
            else:
                yaml.safe_dump(self.to_dict(), fh)


__all__ = [
    "SecurityConfig",
    "P2PConfig", 
    "MeshConfig",
    "DNSConfig",
    "RoutingConfig",
    "PQCConfig",
    "NetworkConfig",
]
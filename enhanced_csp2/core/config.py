from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CoreConfig:
    """Minimal core configuration."""

    node_name: str = "csp-node"
    listen_address: str = "0.0.0.0"
    listen_port: int = 9000
    data_dir: Path = field(default_factory=lambda: Path("./data"))

    enable_p2p: bool = True
    enable_mesh: bool = True
    enable_discovery: bool = True


@dataclass
class FeatureConfig:
    """Optional feature configuration."""

    enable_encryption: bool = False
    enable_advanced_security: bool = False
    enable_compression: bool = False
    enable_batching: bool = False
    enable_ai: bool = False
    enable_quantum: bool = False
    enable_blockchain: bool = False
    enable_storage: bool = False
    enable_metrics: bool = False
    enable_dashboard: bool = False


@dataclass
class NetworkConfig:
    """Complete configuration combining core and features."""

    core: CoreConfig = field(default_factory=CoreConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)

    @classmethod
    def minimal(cls) -> "NetworkConfig":
        return cls(core=CoreConfig(), features=FeatureConfig())

    @classmethod
    def standard(cls) -> "NetworkConfig":
        return cls(
            core=CoreConfig(),
            features=FeatureConfig(
                enable_encryption=True,
                enable_compression=True,
                enable_metrics=True,
            ),
        )

    @classmethod
    def full(cls) -> "NetworkConfig":
        return cls(
            core=CoreConfig(),
            features=FeatureConfig(
                enable_encryption=True,
                enable_advanced_security=True,
                enable_compression=True,
                enable_batching=True,
                enable_ai=True,
                enable_quantum=True,
                enable_blockchain=True,
                enable_storage=True,
                enable_metrics=True,
                enable_dashboard=True,
            ),
        )

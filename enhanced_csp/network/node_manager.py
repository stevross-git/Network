from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from .constants import DEFAULT_STUN_SERVERS, GENESIS_DNS_RECORDS, TLS_ROTATION_DAYS
from .core.config import (
    NetworkConfig,
    SecurityConfig,
    P2PConfig,
    MeshConfig,
    DNSConfig,
    RoutingConfig,
)
from .network_node import EnhancedCSPNetwork

try:
    from enhanced_csp.security_hardening import SecurityOrchestrator
    from enhanced_csp.quantum_csp_engine import QuantumCSPEngine
    from enhanced_csp.blockchain_csp_network import BlockchainCSPNetwork
except Exception:
    # Fallback stub classes
    class SecurityOrchestrator:
        async def initialize(self) -> None:
            pass
        async def shutdown(self) -> None:
            pass
        async def monitor_threats(self) -> None:
            pass
        async def rotate_tls_certificates(self) -> None:
            pass

    class QuantumCSPEngine:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass
        async def initialize(self) -> None:
            pass
        async def shutdown(self) -> None:
            pass

    class BlockchainCSPNetwork:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass
        async def initialize(self) -> None:
            pass
        async def shutdown(self) -> None:
            pass

import logging

logger = logging.getLogger(__name__)


class NodeManager:
    """Manages the lifecycle of an Enhanced CSP node with production features."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.config = self._build_config()
        self.network: Optional[EnhancedCSPNetwork] = None
        self.security_orchestrator: Optional[SecurityOrchestrator] = None
        self.quantum_engine: Optional[QuantumCSPEngine] = None
        self.blockchain: Optional[BlockchainCSPNetwork] = None
        self.shutdown_event = asyncio.Event()
        self.tasks: List[asyncio.Task] = []
        self.logger = self._setup_logging()
        self.is_genesis = args.genesis

    async def initialize(self) -> None:
        self.logger.info("Initializing Enhanced CSP Node...")

        if self.is_genesis:
            self.logger.info("Starting as GENESIS node - First bootstrap and DNS seed")
            await self._initialize_genesis_node()

        self.network = EnhancedCSPNetwork(self.config)

        self.security_orchestrator = SecurityOrchestrator(self.config.security)
        self.logger.info("Initializing security orchestrator")
        await self.security_orchestrator.initialize()

        if self.args.enable_quantum:
            self.quantum_engine = QuantumCSPEngine(self.network)
            await self.quantum_engine.initialize()

        if self.args.enable_blockchain:
            self.blockchain = BlockchainCSPNetwork(self.network)
            await self.blockchain.initialize()

        await self.network.start()

        self.logger.info("Node started with ID: %s", self.network.node_id)

        if self.is_genesis:
            await self._setup_genesis_dns()

        self._start_background_tasks()

    async def _initialize_genesis_node(self) -> None:
        self.logger.info("Configuring genesis node settings...")
        self.config.is_super_peer = True
        self.config.max_peers = 1000
        self.config.p2p.bootstrap_nodes = []
        self.config.node_capabilities = ["relay", "storage", "compute", "dns", "bootstrap"]
        self.config.security.enable_ca_mode = getattr(self.config.security, "enable_ca_mode", True)
        self.config.security.trust_anchors = getattr(self.config.security, "trust_anchors", ["self"])

    async def _setup_genesis_dns(self) -> None:
        self.logger.info("Setting up genesis DNS records...")
        try:
            public_ip = await self._get_public_ip()
            node_multiaddr = f"/ip4/{public_ip}/tcp/{self.config.p2p.listen_port}/p2p/{self.network.node_id}"
            if hasattr(self.network, "dns_overlay") and self.network.dns_overlay:
                for domain, _ in GENESIS_DNS_RECORDS.items():
                    try:
                        if hasattr(self.network.dns_overlay, "register"):
                            await self.network.dns_overlay.register(domain, node_multiaddr)
                            self.logger.info("Registered DNS: %s -> %s", domain, node_multiaddr)
                    except Exception as exc:
                        self.logger.error("Failed to register %s: %s", domain, exc)
                short_id = str(self.network.node_id)[:16]
                try:
                    if hasattr(self.network.dns_overlay, "register"):
                        await self.network.dns_overlay.register(f"{short_id}.web4ai", node_multiaddr)
                except Exception as exc:
                    self.logger.error("Failed to register node ID DNS: %s", exc)
            else:
                self.logger.warning("DNS overlay not available, skipping DNS registration")
        except Exception as exc:
            self.logger.error("Failed to setup genesis DNS: %s", exc)

    async def _get_public_ip(self) -> str:
        if self.config.p2p.stun_servers:
            try:
                import aiostun
                stun_client = aiostun.Client(self.config.p2p.stun_servers[0])
                response = await stun_client.get_external_address()
                return response["external_ip"]
            except Exception:
                pass
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get("https://api.ipify.org") as resp:
                    return await resp.text()
        except Exception:
            if self.config.p2p.listen_address != "0.0.0.0":
                return self.config.p2p.listen_address
            return "127.0.0.1"

    def _start_background_tasks(self) -> None:
        if self.config.security.tls_rotation_interval:
            self.tasks.append(asyncio.create_task(self._tls_rotation_task()))

        self.tasks.append(asyncio.create_task(self._metrics_collection_task()))

        if self.security_orchestrator:
            self.tasks.append(asyncio.create_task(self.security_orchestrator.monitor_threats()))

        if self.is_genesis:
            self.tasks.append(asyncio.create_task(self._genesis_maintenance_task()))

    async def _genesis_maintenance_task(self) -> None:
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(300)
                if hasattr(self.network, "dns_overlay") and self.network.dns_overlay:
                    current_ip = await self._get_public_ip()
                    node_multiaddr = f"/ip4/{current_ip}/tcp/{self.config.p2p.listen_port}/p2p/{self.network.node_id}"
                    for domain in GENESIS_DNS_RECORDS.keys():
                        try:
                            if hasattr(self.network.dns_overlay, "resolve"):
                                existing = await self.network.dns_overlay.resolve(domain)
                                if existing != node_multiaddr:
                                    await self.network.dns_overlay.register(domain, node_multiaddr)
                                    self.logger.info("Updated DNS record: %s", domain)
                        except Exception:
                            pass
                stats = await self.collect_metrics()
                self.logger.info("Genesis node stats: %s peers connected", stats.get("peers_connected", 0))
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.logger.error("Genesis maintenance error: %s", exc)

    async def _tls_rotation_task(self) -> None:
        rotation_interval = timedelta(days=self.args.tls_rotation_days)
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(rotation_interval.total_seconds())
                self.logger.info("Rotating TLS certificates...")
                await self.security_orchestrator.rotate_tls_certificates()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.logger.error("TLS rotation failed: %s", exc)

    async def _metrics_collection_task(self) -> None:
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(60)
                metrics = await self.collect_metrics()
                self.logger.debug(
                    "Metrics: peers=%s, messages_sent=%s, uptime=%ss",
                    metrics.get("peers_connected", 0),
                    metrics.get("messages_sent", 0),
                    metrics.get("uptime", 0),
                )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.logger.error("Metrics collection failed: %s", exc)

    async def collect_metrics(self) -> Dict[str, Any]:
        if self.network:
            return self.network.get_metrics()
        return {
            "peers_connected": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "bandwidth_in": 0,
            "bandwidth_out": 0,
            "uptime": 0,
        }

    async def shutdown(self) -> None:
        self.logger.info("Shutting down Enhanced CSP Node...")
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)

        if self.blockchain:
            await self.blockchain.shutdown()
        if self.quantum_engine:
            await self.quantum_engine.shutdown()
        if self.security_orchestrator:
            self.logger.info("Shutting down security orchestrator")
            await self.security_orchestrator.shutdown()
        if self.network:
            await self.network.stop()
        self.logger.info("Node shutdown complete")

    def _build_config(self) -> NetworkConfig:
        security = SecurityConfig(
            enable_tls=not self.args.no_tls,
            enable_mtls=self.args.mtls,
            enable_pq_crypto=self.args.pq_crypto,
            enable_zero_trust=self.args.zero_trust,
            tls_cert_path=self.args.tls_cert,
            tls_key_path=self.args.tls_key,
            ca_cert_path=self.args.ca_cert,
            audit_log_path=Path(self.args.audit_log) if self.args.audit_log else None,
            enable_threat_detection=not self.args.no_threat_detection,
            enable_intrusion_prevention=self.args.ips,
            enable_compliance_mode=self.args.compliance,
            compliance_standards=self.args.compliance_standards.split(',') if self.args.compliance_standards else [],
            tls_rotation_interval=self.args.tls_rotation_days * 86400,
        )

        p2p = P2PConfig(
            listen_address=self.args.listen_address,
            listen_port=self.args.listen_port,
            bootstrap_nodes=self.args.bootstrap if not self.args.genesis else [],
            stun_servers=self.args.stun_servers or DEFAULT_STUN_SERVERS,
            turn_servers=self.args.turn_servers or [],
            max_peers=self.args.max_peers,
            enable_mdns=not self.args.no_mdns,
        )

        mesh = MeshConfig(
            max_peers=self.args.max_peers,
        )

        dns = DNSConfig(
            root_domain=".web4ai",
        )

        routing = RoutingConfig(
            enable_qos=self.args.qos,
        )

        config = NetworkConfig(
            network_id=self.args.network_id,
            listen_address=self.args.listen_address,
            listen_port=self.args.listen_port,
            node_capabilities=["relay", "storage"] if not self.args.genesis else ["relay", "storage", "compute", "dns", "bootstrap"],
            security=security,
            p2p=p2p,
            mesh=mesh,
            dns=dns,
            routing=routing,
            enable_discovery=True,
            enable_dht=not self.args.no_dht,
            enable_nat_traversal=not self.args.no_nat,
            enable_mesh=True,
            enable_dns=self.args.enable_dns or self.args.genesis,
            enable_adaptive_routing=True,
            enable_metrics=not self.args.no_metrics,
            enable_compression=not self.args.no_compression,
            enable_storage=self.args.enable_storage,
            enable_quantum=self.args.enable_quantum,
            enable_blockchain=self.args.enable_blockchain,
            enable_compute=self.args.enable_compute,
        )

        return config

    def _setup_logging(self) -> logging.Logger:
        log_level = getattr(logging, self.args.log_level.upper())
        logging.basicConfig(
            level=log_level,
            format="%(message)s" if self.args.no_shell else "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )
        return logging.getLogger("enhanced_csp.main")

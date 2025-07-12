from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .core.config import NetworkConfig
from .core.types import NodeID, NodeCapabilities, PeerInfo, NetworkMessage, MessageType

logger = logging.getLogger(__name__)


class SimpleRoutingStub:
    """Fallback routing stub so the node can still start if BatmanRouting is missing."""

    def __init__(self, node: Any = None, topology: Any = None) -> None:
        self.node = node
        self.topology = topology
        self.routing_table: Dict[str, Any] = {}
        self.is_running = False

    async def start(self) -> bool:
        self.is_running = True
        logger.info("Simple routing stub started")
        return True

    async def stop(self) -> None:
        self.is_running = False
        logger.info("Simple routing stub stopped")

    def get_route(self, destination: str) -> Any:
        return self.routing_table.get(destination)

    def get_all_routes(self, destination: str) -> List[Any]:
        route = self.routing_table.get(destination)
        return [route] if route else []


class EnhancedCSPNetwork:
    """Unified Enhanced CSP Network node with optional components."""

    def __init__(self, config: Optional[NetworkConfig] = None) -> None:
        self.config = config or NetworkConfig()
        self.node_id = NodeID.generate()
        self._event_handlers: Dict[str, List[Callable]] = {}

        self.capabilities = NodeCapabilities(
            relay=True,
            storage=self.config.enable_storage,
            compute=self.config.enable_compute,
            quantum=self.config.enable_quantum,
            blockchain=self.config.enable_blockchain,
            dns=self.config.enable_dns,
            bootstrap=False,
        )

        # Core components
        self.transport: Optional[Any] = None
        self.discovery: Optional[Any] = None
        self.dht: Optional[Any] = None
        self.nat: Optional[Any] = None
        self.topology: Optional[Any] = None
        self.routing: Optional[Any] = None
        self.dns_overlay: Optional[Any] = None
        self.adaptive_routing: Optional[Any] = None

        # Runtime state
        self.peers: Dict[NodeID, PeerInfo] = {}
        self.is_running = False
        self._message_handlers: Dict[MessageType, List[Callable]] = {}
        self._background_tasks: List[asyncio.Task] = []

        # Stats/metrics
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
            "peers_connected": 0,
            "start_time": None,
        }
        self.metrics = {
            "messages_sent": 0,
            "messages_received": 0,
            "peers_connected": 0,
            "bandwidth_in": 0,
            "bandwidth_out": 0,
            "routing_table_size": 0,
            "uptime": 0,
            "last_updated": time.time(),
        }

    def on_event(self, name: str, handler: Callable[[dict], Any]) -> None:
        self._event_handlers.setdefault(name, []).append(handler)

    async def _dispatch_event(self, name: str, payload: dict) -> None:
        for fn in self._event_handlers.get(name, []):
            try:
                if asyncio.iscoroutinefunction(fn):
                    await fn(payload)
                else:
                    fn(payload)
            except Exception:
                logger.exception("Unhandled exception in %s handler", name)

    async def send_message(self, *args) -> bool:
        if not self.transport or not self.is_running:
            logger.error("Transport not initialised or node not running")
            return False

        try:
            if len(args) == 1 and isinstance(args[0], NetworkMessage):
                nm: NetworkMessage = args[0]
                packet = {
                    "type": nm.type.value if hasattr(nm.type, "value") else str(nm.type),
                    "payload": nm.payload,
                    "sender": str(nm.sender),
                    "recipient": str(nm.recipient) if nm.recipient else None,
                    "timestamp": nm.timestamp.isoformat() if hasattr(nm.timestamp, "isoformat") else str(nm.timestamp),
                }
                peer = str(nm.recipient)
                success = await self.transport.send(peer, packet)
            elif len(args) == 2:
                peer, packet = args
                success = await self.transport.send(str(peer), packet)
            else:
                raise ValueError("Invalid arguments for send_message")

            if success:
                self.stats["messages_sent"] += 1
                self.stats["bytes_sent"] += len(str(packet))
                self.metrics["messages_sent"] += 1
                self.metrics["last_updated"] = time.time()
            return success
        except Exception as exc:
            logger.exception("Failed to send message: %s", exc)
            return False

    async def start(self) -> bool:
        if self.is_running:
            logger.warning("Node is already running")
            return True

        try:
            logger.info("Starting Enhanced CSP network node %s", self.node_id)
            await self._initialize_components()

            if self.transport and not await self._start_transport():
                return False
            if self.discovery and not await self._start_discovery():
                return False
            if self.dht and not await self._start_dht():
                return False
            if self.nat and not await self._start_nat():
                return False
            if self.topology and not await self._start_topology():
                return False
            if self.routing and not await self._start_routing():
                return False
            if self.dns_overlay and not await self._start_dns():
                return False
            if self.adaptive_routing and not await self._start_adaptive_routing():
                return False

            self._start_background_tasks()
            self.is_running = True
            self.stats["start_time"] = time.time()
            logger.info("Enhanced CSP network node %s started successfully", self.node_id)
            return True
        except Exception:
            logger.exception("Failed to start network node")
            await self.stop()
            return False

    async def _initialize_components(self) -> None:
        try:
            from ..p2p.transport import MultiProtocolTransport
            self.transport = MultiProtocolTransport(self.config.p2p)
            from ..p2p.discovery import HybridDiscovery
            self.discovery = HybridDiscovery(self.config.p2p, self.node_id)

            if self.config.enable_dht:
                from ..p2p.dht import KademliaDHT
                self.dht = KademliaDHT(self.node_id, self.transport)

            from ..p2p.nat import NATTraversal
            self.nat = NATTraversal(self.config.p2p)

            if self.config.enable_mesh:
                from ..mesh.topology import MeshTopologyManager

                async def _send(peer: str, pkt: Any) -> bool:
                    return await self.transport.send(peer, pkt)

                self.topology = MeshTopologyManager(self.node_id, self.config.mesh, _send)

            if getattr(self.config, "enable_routing", True) and self.topology:
                try:
                    from ..mesh.routing import BatmanRouting
                    self.routing = BatmanRouting(self, self.topology)
                    logger.info("BatmanRouting initialised successfully")
                except Exception as exc:
                    logger.warning("BatmanRouting unavailable (%s) – using stub", exc)
                    self.routing = SimpleRoutingStub(self, self.topology)

            if self.config.enable_dns:
                from ..dns.overlay import DNSOverlay
                self.dns_overlay = DNSOverlay(self)

            if self.config.enable_adaptive_routing and self.routing:
                from ..routing.adaptive import AdaptiveRoutingEngine
                self.adaptive_routing = AdaptiveRoutingEngine(self, self.config.routing, self.routing)

        except ImportError as e:
            logger.warning("Some components unavailable: %s. Using stubs where possible.", e)

            if not self.transport:
                class StubTransport:
                    def __init__(self) -> None:
                        self.is_running = True
                        self.handlers = {}

                    async def start(self) -> bool:
                        logger.info("Stub transport started")
                        return True

                    async def stop(self) -> bool:
                        logger.info("Stub transport stopped")
                        return True

                    async def send(self, peer: str, data: Any) -> bool:
                        logger.debug(f"Stub transport: would send to {peer}: {data}")
                        return False

                    def register_handler(self, event: str, handler: Callable) -> None:
                        self.handlers[event] = handler
                        logger.debug(f"Registered handler for {event}")

                self.transport = StubTransport()

            if not self.discovery:
                class StubDiscovery:
                    def __init__(self) -> None:
                        self.is_running = True

                    async def start(self) -> bool:
                        logger.info("Stub discovery started")
                        return True

                    async def stop(self) -> bool:
                        logger.info("Stub discovery stopped")
                        return True

                self.discovery = StubDiscovery()

            if not self.nat:
                class StubNAT:
                    def __init__(self) -> None:
                        self.is_running = True

                    async def start(self) -> bool:
                        logger.info("Stub NAT traversal started")
                        return True

                    async def stop(self) -> bool:
                        logger.info("Stub NAT traversal stopped")
                        return True

                self.nat = StubNAT()

            if not self.topology:
                class StubTopology:
                    def __init__(self) -> None:
                        self.is_running = True

                    async def start(self) -> bool:
                        logger.info("Stub topology manager started")
                        return True

                    async def stop(self) -> bool:
                        logger.info("Stub topology manager stopped")
                        return True

                self.topology = StubTopology()

            if not self.routing:
                self.routing = SimpleRoutingStub(self, self.topology)

            if not self.dns_overlay:
                class StubDNS:
                    def __init__(self) -> None:
                        self.is_running = True

                    async def start(self) -> bool:
                        logger.info("Stub DNS overlay started")
                        return True

                    async def stop(self) -> bool:
                        logger.info("Stub DNS overlay stopped")
                        return True

                    async def register(self, *args: Any, **kwargs: Any) -> None:
                        logger.debug("Stub DNS register called")

                    async def resolve(self, *args: Any, **kwargs: Any) -> str:
                        logger.debug("Stub DNS resolve called")
                        return ""

                    async def list_records(self) -> Dict[str, str]:
                        return {}

                self.dns_overlay = StubDNS()

            if not self.adaptive_routing:
                class StubAR:
                    def __init__(self) -> None:
                        self.is_running = True

                    async def start(self) -> bool:
                        logger.info("Stub adaptive routing started")
                        return True

                    async def stop(self) -> bool:
                        logger.info("Stub adaptive routing stopped")
                        return True

                self.adaptive_routing = StubAR()

    async def _start_transport(self) -> bool:
        try:
            if hasattr(self.transport, "start"):
                return await self.transport.start()
            return True
        except Exception as e:
            logger.exception("Failed to start transport: %s", e)
            return False

    async def _start_discovery(self) -> bool:
        try:
            if hasattr(self.discovery, "start"):
                await self.discovery.start()
            return True
        except Exception as e:
            logger.exception("Failed to start discovery: %s", e)
            return False

    async def _start_dht(self) -> bool:
        try:
            if hasattr(self.dht, "start"):
                await self.dht.start()
            return True
        except Exception as e:
            logger.exception("Failed to start DHT: %s", e)
            return False

    async def _start_nat(self) -> bool:
        if not self.nat:
            return True
        try:
            if hasattr(self.nat, "start"):
                await self.nat.start()
            return True
        except Exception as e:
            logger.exception("Failed to start NAT traversal: %s", e)
            return False

    async def _start_topology(self) -> bool:
        try:
            if hasattr(self.topology, "start"):
                await self.topology.start()
            return True
        except Exception as e:
            logger.exception("Failed to start topology manager: %s", e)
            return False

    async def _start_routing(self) -> bool:
        if not self.routing:
            return True
        try:
            if hasattr(self.routing, "start"):
                return await self.routing.start()
            return True
        except Exception as e:
            logger.exception("Failed to start routing layer: %s", e)
            return False

    async def _start_dns(self) -> bool:
        if not self.dns_overlay:
            return True
        try:
            if hasattr(self.dns_overlay, "start"):
                await self.dns_overlay.start()
            return True
        except Exception as e:
            logger.exception("Failed to start DNS overlay: %s", e)
            return False

    async def _start_adaptive_routing(self) -> bool:
        if not self.adaptive_routing:
            return True
        try:
            if hasattr(self.adaptive_routing, "start"):
                await self.adaptive_routing.start()
            return True
        except Exception as e:
            logger.exception("Failed to start adaptive routing: %s", e)
            return False

    async def stop(self) -> bool:
        if not self.is_running:
            return True

        for task in self._background_tasks:
            task.cancel()
        await asyncio.gather(*self._background_tasks, return_exceptions=True)

        for component in (
            self.adaptive_routing,
            self.dns_overlay,
            self.routing,
            self.topology,
            self.nat,
            self.dht,
            self.discovery,
            self.transport,
        ):
            if component and hasattr(component, "stop"):
                try:
                    await component.stop()
                except Exception:
                    logger.exception("Error stopping %s", component)

        self.is_running = False
        logger.info("Enhanced CSP network node %s stopped", self.node_id)
        return True

    async def broadcast_message(self, message: NetworkMessage) -> int:
        if not self.is_running or not self.transport:
            return 0
        packet = {
            "type": message.type.value if hasattr(message.type, "value") else str(message.type),
            "payload": message.payload,
            "sender": str(message.sender),
            "timestamp": message.timestamp.isoformat() if hasattr(message.timestamp, "isoformat") else str(message.timestamp),
        }
        count = 0
        for pid in self.peers:
            if hasattr(self.transport, "send") and await self.transport.send(str(pid), packet):
                count += 1
        if count:
            self.stats["messages_sent"] += count
            self.stats["bytes_sent"] += len(str(packet)) * count
            self.metrics["messages_sent"] += count
            self.metrics["last_updated"] = time.time()
        return count

    def _start_background_tasks(self) -> None:
        self._background_tasks = [
            asyncio.create_task(self._peer_maintenance_loop()),
            asyncio.create_task(self._stats_update_loop()),
            asyncio.create_task(self._health_check_loop()),
        ]

    async def _peer_maintenance_loop(self) -> None:
        while self.is_running:
            try:
                now = datetime.utcnow()
                stale = [
                    pid for pid, info in self.peers.items()
                    if hasattr(info, "last_seen") and (now - info.last_seen).total_seconds() > 300
                ]
                for pid in stale:
                    del self.peers[pid]

                self.stats["peers_connected"] = len(self.peers)
                self.metrics["peers_connected"] = len(self.peers)
                self.metrics["routing_table_size"] = len(self.routing.routing_table) if self.routing and hasattr(self.routing, "routing_table") else 0
                self.metrics["uptime"] = time.time() - self.stats["start_time"] if self.stats["start_time"] else 0
                self.metrics["last_updated"] = time.time()

                await asyncio.sleep(60)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Peer maintenance error")
                await asyncio.sleep(60)

    async def _stats_update_loop(self) -> None:
        while self.is_running:
            try:
                if self.stats["start_time"]:
                    uptime = time.time() - self.stats["start_time"]
                    logger.info(
                        "Node stats – Uptime: %.0fs, Sent: %d, Received: %d, Peers: %d",
                        uptime,
                        self.stats["messages_sent"],
                        self.stats["messages_received"],
                        self.stats["peers_connected"],
                    )
                await asyncio.sleep(300)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Stats loop error")
                await asyncio.sleep(300)

    async def _health_check_loop(self) -> None:
        while self.is_running:
            try:
                healthy = True
                if self.transport and not getattr(self.transport, "is_running", True):
                    healthy = False
                if self.discovery and not getattr(self.discovery, "is_running", True):
                    healthy = False
                if not healthy:
                    logger.warning("Node health check failed")
                await asyncio.sleep(120)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Health check error")
                await asyncio.sleep(120)

    def add_message_handler(self, message_type: MessageType, handler: Callable) -> None:
        self._message_handlers.setdefault(message_type, []).append(handler)
        if self.transport and hasattr(self.transport, "register_handler"):
            self.transport.register_handler(str(message_type), handler)

    def remove_message_handler(self, message_type: MessageType, handler: Callable) -> None:
        self._message_handlers.get(message_type, []).remove(handler)

    def get_stats(self) -> Dict[str, Any]:
        return self.stats.copy()

    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics.copy()

    def get_peer_info(self, peer_id: NodeID) -> Optional[PeerInfo]:
        return self.peers.get(peer_id)

    def get_all_peers(self) -> Dict[NodeID, PeerInfo]:
        return self.peers.copy()

    def get_peers(self) -> List[PeerInfo]:
        return list(self.peers.values())

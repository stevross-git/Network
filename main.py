#!/usr/bin/env python3
"""
Enhanced CSP Network - Comprehensive Main Entry Point
Full-featured network node with all capabilities enabled.

Usage:
    python main.py --genesis                          # Start as genesis node
    python main.py --connect genesis.peoplesainetwork.com  # Connect to network
    python main.py --benchmark                        # Run performance tests
    python main.py --dashboard                        # Start with web dashboard
"""

import asyncio
import sys
import signal
import logging
import argparse
import json
import time
import threading
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable
from datetime import datetime
import traceback
import socket
import psutil

def setup_python_path():
    """Setup Python path to find enhanced_csp modules."""
    script_dir = Path(__file__).resolve().parent
    project_root = None
    
    # Find project root
    for candidate in [script_dir, script_dir.parent]:
        if (candidate / "enhanced_csp").exists():
            project_root = candidate
            break
    
    if project_root is None:
        cwd = Path.cwd()
        if (cwd / "enhanced_csp").exists():
            project_root = cwd
        else:
            print("❌ Could not find enhanced_csp directory!")
            return None
    
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    return project_root

# Setup path and imports
project_root = setup_python_path()
if project_root is None:
    sys.exit(1)

try:
    # Core imports
    from enhanced_csp.network.core.config import (
        NetworkConfig, P2PConfig, MeshConfig, SecurityConfig, 
        DNSConfig, RoutingConfig
    )
    from enhanced_csp.network.core.types import (
        NodeID, PeerInfo, NetworkMessage, MessageType, NodeCapabilities
    )
    from enhanced_csp.network.core.node import NetworkNode
    
    # Utility imports
    from enhanced_csp.network.utils import (
        get_logger, setup_logging, validate_ip_address, 
        validate_port_number, format_bytes, format_duration
    )
    
    # P2P and networking
    from enhanced_csp.network.p2p.discovery import HybridDiscovery, PeerExchange
    from enhanced_csp.network.p2p.transport import P2PTransport, MultiProtocolTransport
    from enhanced_csp.network.p2p.nat import NATTraversal
    
    # Mesh networking
    from enhanced_csp.network.mesh.topology import MeshTopologyManager
    from enhanced_csp.network.mesh.routing import BatmanRouting
    
    # DNS and routing
    from enhanced_csp.network.dns.overlay import DNSOverlay
    from enhanced_csp.network.routing.adaptive import AdaptiveRoutingEngine
    
    # Security
    from enhanced_csp.network.security.security_hardening import SecurityOrchestrator
    
    # Performance optimizations
    from enhanced_csp.network.optimization.batching import IntelligentBatching
    from enhanced_csp.network.optimization.compression import AdaptiveCompression
    from enhanced_csp.network.optimization.protocol_optimizer import ProtocolOptimizer
    
    IMPORTS_AVAILABLE = True
    
except ImportError as e:
    print(f"⚠️  Some Enhanced CSP imports failed: {e}")
    print("🔧 Running with limited functionality...")
    IMPORTS_AVAILABLE = False


class EnhancedCSPMain:
    """
    Comprehensive Enhanced CSP Network node with all features.
    """
    
    def __init__(self):
        # Core components
        self.node: Optional[NetworkNode] = None
        self.config: Optional[NetworkConfig] = None
        
        # State management
        self.running = False
        self.start_time = time.time()
        self.shutdown_event = asyncio.Event()
        
        # Performance monitoring
        self.metrics = {
            'messages_sent': 0,
            'messages_received': 0,
            'bytes_sent': 0,
            'bytes_received': 0,
            'peers_connected': 0,
            'uptime_seconds': 0,
            'cpu_usage': 0.0,
            'memory_usage': 0.0
        }
        
        # Component managers
        self.security_orchestrator: Optional[SecurityOrchestrator] = None
        self.performance_optimizer: Optional[ProtocolOptimizer] = None
        self.adaptive_compression: Optional[AdaptiveCompression] = None
        self.intelligent_batching: Optional[IntelligentBatching] = None
        
        # Logging
        self.logger = logging.getLogger('enhanced_csp_main')
        
        # Dashboard and monitoring
        self.dashboard_enabled = False
        self.dashboard_port = 8080
        self.status_server = None
        
        # Event handlers
        self.message_handlers: Dict[str, Callable] = {}
        self.peer_handlers: Dict[str, Callable] = {}
        
    def create_comprehensive_config(self, args) -> NetworkConfig:
        """Create a comprehensive network configuration with all features."""
        
        if not IMPORTS_AVAILABLE:
            return self._create_minimal_config(args)
        
        config = NetworkConfig()
        
        # === P2P Configuration ===
        config.p2p = P2PConfig()
        config.p2p.listen_address = args.listen_address
        config.p2p.listen_port = args.port
        config.p2p.enable_mdns = args.enable_mdns
        config.p2p.enable_upnp = args.enable_upnp
        config.p2p.enable_nat_traversal = args.enable_nat
        config.p2p.enable_quic = args.enable_quic
        config.p2p.max_connections = args.max_connections
        config.p2p.connection_timeout = args.connection_timeout
        config.p2p.max_message_size = args.max_message_size
        
        # Bootstrap nodes
        if args.genesis_host and not args.genesis:
            genesis_addr = f"/ip4/{args.genesis_host}/tcp/{args.genesis_port}"
            config.p2p.bootstrap_nodes = [genesis_addr]
        
        if args.bootstrap_nodes:
            config.p2p.bootstrap_nodes.extend(args.bootstrap_nodes)
        
        # DNS seed configuration
        if args.dns_seed:
            config.p2p.dns_seed_domain = args.dns_seed
        
        # === Mesh Configuration ===
        config.mesh = MeshConfig()
        config.mesh.topology_type = args.mesh_topology
        config.mesh.enable_super_peers = args.enable_super_peers
        config.mesh.max_peers = args.max_peers
        config.mesh.routing_update_interval = args.routing_interval
        config.mesh.link_quality_threshold = args.link_quality_threshold
        config.mesh.enable_multi_hop = args.enable_multi_hop
        config.mesh.max_hop_count = args.max_hops
        
        # === Security Configuration ===
        config.security = SecurityConfig()
        config.security.enable_encryption = args.enable_encryption
        config.security.enable_authentication = args.enable_auth
        config.security.enable_tls = args.enable_tls
        config.security.key_size = args.key_size
        config.security.cipher_suite = args.cipher_suite
        
        # === DNS Configuration ===
        config.dns = DNSConfig()
        config.dns.enable_dnssec = args.enable_dnssec
        config.dns.root_domain = args.dns_root_domain
        config.dns.default_ttl = args.dns_ttl
        config.dns.cache_size = args.dns_cache_size
        
        # === Routing Configuration ===
        config.routing = RoutingConfig()
        config.routing.enable_multipath = args.enable_multipath
        config.routing.enable_ml_predictor = args.enable_ml_routing
        config.routing.max_paths_per_destination = args.max_paths
        config.routing.failover_threshold_ms = args.failover_threshold
        config.routing.enable_congestion_control = args.enable_congestion_control
        config.routing.enable_qos = args.enable_qos
        
        # === Node Configuration ===
        config.node_name = args.node_name
        config.node_type = "genesis" if args.genesis else "standard"
        
        # Node capabilities
        config.capabilities = NodeCapabilities()
        config.capabilities.can_relay = args.enable_relay
        config.capabilities.can_store = args.enable_storage
        config.capabilities.can_compute = args.enable_compute
        config.capabilities.max_storage_mb = args.max_storage
        config.capabilities.max_compute_units = args.max_compute
        
        # === Feature Flags ===
        config.enable_dht = args.enable_dht
        config.enable_mesh = args.enable_mesh
        config.enable_dns = args.enable_dns_overlay
        config.enable_adaptive_routing = args.enable_adaptive_routing
        config.enable_optimizations = args.enable_optimizations
        config.enable_dashboard = args.enable_dashboard
        
        # === Data and Logging ===
        config.data_dir = Path(args.data_dir)
        config.data_dir.mkdir(parents=True, exist_ok=True)
        
        config.log_level = args.log_level
        config.log_file = args.log_file
        
        self.logger.info(f"🔧 Comprehensive configuration created")
        self.logger.info(f"   🌐 Listen: {config.p2p.listen_address}:{config.p2p.listen_port}")
        self.logger.info(f"   🔗 Bootstrap nodes: {len(config.p2p.bootstrap_nodes)}")
        self.logger.info(f"   🏷️  Node type: {config.node_type}")
        self.logger.info(f"   🎯 Features: DHT={config.enable_dht}, Mesh={config.enable_mesh}")
        
        return config
    
    def _create_minimal_config(self, args):
        """Create minimal config when imports are limited."""
        class MinimalConfig:
            def __init__(self):
                self.node_name = args.node_name
                self.listen_port = args.port
                self.genesis_host = getattr(args, 'genesis_host', None)
                self.genesis_port = getattr(args, 'genesis_port', 30300)
                self.data_dir = Path(args.data_dir)
        
        return MinimalConfig()
    
    async def initialize_components(self):
        """Initialize all network components and optimizations."""
        if not IMPORTS_AVAILABLE or not self.node:
            self.logger.warning("⚠️  Limited initialization - imports not available")
            return
        
        try:
            self.logger.info("🔧 Initializing network components...")
            
            # === Security Orchestrator ===
            if hasattr(self.config, 'security') and self.config.security.enable_encryption:
                self.security_orchestrator = SecurityOrchestrator(self.config)
                await self.security_orchestrator.initialize()
                self.logger.info("   🔒 Security orchestrator initialized")
            
            # === Performance Optimizations ===
            if getattr(self.config, 'enable_optimizations', False):
                # Protocol optimizer
                self.performance_optimizer = ProtocolOptimizer()
                await self.performance_optimizer.initialize()
                
                # Adaptive compression
                self.adaptive_compression = AdaptiveCompression()
                await self.adaptive_compression.initialize()
                
                # Intelligent batching
                self.intelligent_batching = IntelligentBatching()
                await self.intelligent_batching.initialize()
                
                self.logger.info("   ⚡ Performance optimizations initialized")
            
            # === Message Handlers ===
            self.setup_message_handlers()
            
            # === Peer Event Handlers ===
            self.setup_peer_handlers()
            
            # === Dashboard ===
            if getattr(self.config, 'enable_dashboard', False):
                await self.start_dashboard()
            
            self.logger.info("✅ All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Component initialization failed: {e}")
            raise
    
    def setup_message_handlers(self):
        """Setup message handlers for different message types."""
        self.message_handlers = {
            'ping': self.handle_ping_message,
            'pong': self.handle_pong_message,
            'data': self.handle_data_message,
            'control': self.handle_control_message,
            'discovery': self.handle_discovery_message,
            'routing': self.handle_routing_message,
            'peer_exchange': self.handle_peer_exchange_message,
            'heartbeat': self.handle_heartbeat_message,
            'status_request': self.handle_status_request,
            'benchmark': self.handle_benchmark_message
        }
        
        self.logger.info(f"📨 Registered {len(self.message_handlers)} message handlers")
    
    def setup_peer_handlers(self):
        """Setup peer event handlers."""
        self.peer_handlers = {
            'peer_connected': self.handle_peer_connected,
            'peer_disconnected': self.handle_peer_disconnected,
            'peer_discovered': self.handle_peer_discovered,
            'peer_quality_changed': self.handle_peer_quality_changed
        }
        
        self.logger.info(f"👥 Registered {len(self.peer_handlers)} peer handlers")
    
    async def start_node(self) -> bool:
        """Start the Enhanced CSP Network node with full features."""
        if not IMPORTS_AVAILABLE:
            return await self.start_minimal_node()
        
        try:
            self.logger.info("🚀 Starting Enhanced CSP Network Node...")
            self.logger.info(f"🏷️  Node: {self.config.node_name}")
            self.logger.info(f"🌐 Type: {self.config.node_type}")
            
            # Create and start the network node
            self.node = NetworkNode(self.config)
            
            # Hook into node events
            if hasattr(self.node, 'on_message_received'):
                self.node.on_message_received = self.handle_incoming_message
            if hasattr(self.node, 'on_peer_connected'):
                self.node.on_peer_connected = self.handle_peer_connected
            if hasattr(self.node, 'on_peer_disconnected'):
                self.node.on_peer_disconnected = self.handle_peer_disconnected
            
            await self.node.start()
            
            # Initialize additional components
            await self.initialize_components()
            
            self.running = True
            self.start_time = time.time()
            
            self.logger.info("✅ Network node started successfully!")
            self.logger.info(f"🆔 Node ID: {self.node.node_id}")
            self.logger.info(f"📊 Capabilities: {self.node.capabilities}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start network node: {e}")
            traceback.print_exc()
            return False
    
    async def start_minimal_node(self) -> bool:
        """Start node in minimal mode when imports are limited."""
        self.logger.info("🔧 Starting node in minimal mode...")
        self.running = True
        self.start_time = time.time()
        
        # Simulate node startup
        await asyncio.sleep(1)
        self.logger.info("✅ Minimal node started")
        return True
    
    async def connect_to_network(self) -> bool:
        """Connect to the Enhanced CSP Network."""
        if not self.node or not IMPORTS_AVAILABLE:
            self.logger.info("🔧 Simulating network connection...")
            await asyncio.sleep(2)
            self.logger.info("✅ Mock network connection established")
            return True
        
        try:
            self.logger.info("🔍 Connecting to Enhanced CSP Network...")
            
            # Attempt to discover and connect to peers
            max_attempts = 30
            attempt = 0
            connected_peers = 0
            
            while attempt < max_attempts and self.running:
                try:
                    # Check discovery status
                    if hasattr(self.node, 'discovery') and self.node.discovery:
                        peers = await self.node.discovery.find_peers(count=10)
                        if peers:
                            self.logger.info(f"🎉 Discovered {len(peers)} peers!")
                            for peer in peers[:3]:  # Log first 3
                                peer_id = peer.get('node_id', 'unknown')[:16]
                                source = peer.get('source', 'unknown')
                                self.logger.info(f"   📡 {peer_id}... (via {source})")
                            
                            connected_peers = len(peers)
                            break
                    
                    # Check direct peer connections
                    if hasattr(self.node, 'peers'):
                        connected_peers = len(self.node.peers)
                        if connected_peers > 0:
                            self.logger.info(f"🌐 Connected to {connected_peers} peers")
                            break
                    
                except Exception as e:
                    self.logger.debug(f"Connection attempt {attempt + 1} error: {e}")
                
                attempt += 1
                await asyncio.sleep(2)
                
                if attempt % 5 == 0:
                    self.logger.info(f"🔄 Connecting... (attempt {attempt}/{max_attempts})")
            
            if connected_peers > 0:
                self.logger.info(f"✅ Successfully connected to network!")
                self.logger.info(f"👥 Connected peers: {connected_peers}")
                return True
            else:
                self.logger.warning("⚠️  Could not connect to network within timeout")
                self.logger.info("💡 Node will continue running and retry connections")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Network connection failed: {e}")
            return False
    
    async def run_main_event_loop(self):
        """Main event loop with comprehensive network operations."""
        self.logger.info("🔄 Starting main event loop...")
        
        # Start background tasks
        tasks = []
        
        if self.running:
            tasks.extend([
                asyncio.create_task(self.metrics_collection_loop()),
                asyncio.create_task(self.peer_maintenance_loop()),
                asyncio.create_task(self.network_optimization_loop()),
                asyncio.create_task(self.status_reporting_loop()),
                asyncio.create_task(self.health_monitoring_loop())
            ])
        
        try:
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
        except Exception as e:
            self.logger.error(f"❌ Main loop error: {e}")
        finally:
            # Cancel all tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            self.logger.info("🛑 Main event loop stopped")
    
    async def metrics_collection_loop(self):
        """Collect and update performance metrics."""
        while self.running:
            try:
                # Update basic metrics
                self.metrics['uptime_seconds'] = time.time() - self.start_time
                self.metrics['cpu_usage'] = psutil.cpu_percent()
                self.metrics['memory_usage'] = psutil.Process().memory_info().rss / (1024 * 1024)
                
                # Update network metrics
                if self.node and hasattr(self.node, 'peers'):
                    self.metrics['peers_connected'] = len(self.node.peers)
                
                # Update throughput metrics (from node if available)
                if self.node and hasattr(self.node, 'metrics'):
                    node_metrics = self.node.metrics
                    self.metrics['messages_sent'] = getattr(node_metrics, 'messages_sent', 0)
                    self.metrics['messages_received'] = getattr(node_metrics, 'messages_received', 0)
                    self.metrics['bytes_sent'] = getattr(node_metrics, 'bytes_sent', 0)
                    self.metrics['bytes_received'] = getattr(node_metrics, 'bytes_received', 0)
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.debug(f"Metrics collection error: {e}")
                await asyncio.sleep(10)
    
    async def peer_maintenance_loop(self):
        """Maintain peer connections and quality."""
        while self.running:
            try:
                if self.node and hasattr(self.node, 'discovery'):
                    # Trigger peer discovery
                    await self.node.discovery.find_peers(count=5)
                
                # Maintain optimal peer count
                if self.node and hasattr(self.node, 'peers'):
                    peer_count = len(self.node.peers)
                    max_peers = getattr(self.config, 'max_peers', 50)
                    
                    if peer_count < max_peers // 2:
                        self.logger.info(f"🔍 Low peer count ({peer_count}), discovering more...")
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.debug(f"Peer maintenance error: {e}")
                await asyncio.sleep(60)
    
    async def network_optimization_loop(self):
        """Continuously optimize network performance."""
        while self.running:
            try:
                # Apply performance optimizations
                if self.performance_optimizer:
                    await self.performance_optimizer.optimize()
                
                if self.adaptive_compression:
                    await self.adaptive_compression.adapt()
                
                if self.intelligent_batching:
                    await self.intelligent_batching.optimize_batches()
                
                await asyncio.sleep(120)  # Optimize every 2 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.debug(f"Network optimization error: {e}")
                await asyncio.sleep(120)
    
    async def status_reporting_loop(self):
        """Periodic status reporting."""
        while self.running:
            try:
                await asyncio.sleep(300)  # Report every 5 minutes
                await self.print_comprehensive_status()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.debug(f"Status reporting error: {e}")
                await asyncio.sleep(300)
    
    async def health_monitoring_loop(self):
        """Monitor node health and auto-recovery."""
        while self.running:
            try:
                # Check system health
                cpu_usage = self.metrics['cpu_usage']
                memory_usage = self.metrics['memory_usage']
                peer_count = self.metrics['peers_connected']
                
                # Health warnings
                if cpu_usage > 90:
                    self.logger.warning(f"⚠️  High CPU usage: {cpu_usage:.1f}%")
                
                if memory_usage > 1000:  # 1GB
                    self.logger.warning(f"⚠️  High memory usage: {memory_usage:.1f}MB")
                
                if peer_count == 0:
                    self.logger.warning("⚠️  No peers connected - attempting recovery")
                    # Trigger reconnection
                    if self.node and hasattr(self.node, 'discovery'):
                        asyncio.create_task(self.connect_to_network())
                
                await asyncio.sleep(180)  # Check every 3 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.debug(f"Health monitoring error: {e}")
                await asyncio.sleep(180)
    
    # === Message Handlers ===
    
    async def handle_incoming_message(self, peer_id: str, message: Any):
        """Handle incoming messages from peers."""
        try:
            message_type = getattr(message, 'type', 'unknown')
            
            if message_type in self.message_handlers:
                await self.message_handlers[message_type](peer_id, message)
            else:
                self.logger.debug(f"📨 Unhandled message type: {message_type}")
            
            # Update metrics
            self.metrics['messages_received'] += 1
            
        except Exception as e:
            self.logger.error(f"❌ Message handling error: {e}")
    
    async def handle_ping_message(self, peer_id: str, message: Any):
        """Handle ping messages."""
        self.logger.debug(f"🏓 Ping from {peer_id[:16]}...")
        
        # Send pong response
        if self.node:
            await self.send_message(peer_id, {'type': 'pong', 'timestamp': time.time()})
    
    async def handle_pong_message(self, peer_id: str, message: Any):
        """Handle pong messages."""
        timestamp = getattr(message, 'timestamp', time.time())
        latency = (time.time() - timestamp) * 1000  # ms
        self.logger.debug(f"🏓 Pong from {peer_id[:16]}... (latency: {latency:.2f}ms)")
    
    async def handle_data_message(self, peer_id: str, message: Any):
        """Handle data messages."""
        data_size = len(getattr(message, 'data', b''))
        self.logger.debug(f"📊 Data message from {peer_id[:16]}... ({data_size} bytes)")
        
        # Update metrics
        self.metrics['bytes_received'] += data_size
    
    async def handle_control_message(self, peer_id: str, message: Any):
        """Handle control messages."""
        command = getattr(message, 'command', 'unknown')
        self.logger.debug(f"🎛️  Control message from {peer_id[:16]}...: {command}")
    
    async def handle_discovery_message(self, peer_id: str, message: Any):
        """Handle peer discovery messages."""
        self.logger.debug(f"🔍 Discovery message from {peer_id[:16]}...")
    
    async def handle_routing_message(self, peer_id: str, message: Any):
        """Handle routing protocol messages."""
        self.logger.debug(f"🗺️  Routing message from {peer_id[:16]}...")
    
    async def handle_peer_exchange_message(self, peer_id: str, message: Any):
        """Handle peer exchange messages."""
        peers = getattr(message, 'peers', [])
        self.logger.debug(f"👥 Peer exchange from {peer_id[:16]}... ({len(peers)} peers)")
    
    async def handle_heartbeat_message(self, peer_id: str, message: Any):
        """Handle heartbeat messages."""
        self.logger.debug(f"💓 Heartbeat from {peer_id[:16]}...")
    
    async def handle_status_request(self, peer_id: str, message: Any):
        """Handle status request messages."""
        status = await self.get_node_status()
        await self.send_message(peer_id, {'type': 'status_response', 'status': status})
    
    async def handle_benchmark_message(self, peer_id: str, message: Any):
        """Handle benchmark messages."""
        self.logger.debug(f"📊 Benchmark message from {peer_id[:16]}...")
        
        # Echo back for latency measurement
        await self.send_message(peer_id, {
            'type': 'benchmark_response', 
            'original_timestamp': getattr(message, 'timestamp', time.time()),
            'response_timestamp': time.time()
        })
    
    # === Peer Event Handlers ===
    
    async def handle_peer_connected(self, peer_id: str, peer_info: Any):
        """Handle peer connection events."""
        self.logger.info(f"👋 Peer connected: {peer_id[:16]}...")
        self.metrics['peers_connected'] = getattr(self.node, 'peer_count', 0)
    
    async def handle_peer_disconnected(self, peer_id: str, reason: str = "unknown"):
        """Handle peer disconnection events."""
        self.logger.info(f"👋 Peer disconnected: {peer_id[:16]}... (reason: {reason})")
        self.metrics['peers_connected'] = getattr(self.node, 'peer_count', 0)
    
    async def handle_peer_discovered(self, peer_info: Dict[str, Any]):
        """Handle peer discovery events."""
        peer_id = peer_info.get('node_id', 'unknown')
        source = peer_info.get('source', 'unknown')
        self.logger.debug(f"🔍 Peer discovered: {peer_id[:16]}... (via {source})")
    
    async def handle_peer_quality_changed(self, peer_id: str, quality: float):
        """Handle peer quality changes."""
        self.logger.debug(f"📊 Peer quality changed: {peer_id[:16]}... (quality: {quality:.2f})")
    
    # === Utility Methods ===
    
    async def send_message(self, peer_id: str, message: Dict[str, Any]) -> bool:
        """Send a message to a peer."""
        try:
            if self.node and hasattr(self.node, 'send_message'):
                success = await self.node.send_message(peer_id, message)
                if success:
                    self.metrics['messages_sent'] += 1
                    self.metrics['bytes_sent'] += len(json.dumps(message).encode())
                return success
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Send message error: {e}")
            return False
    
    async def broadcast_message(self, message: Dict[str, Any], exclude_peers: List[str] = None) -> int:
        """Broadcast a message to all connected peers."""
        exclude_peers = exclude_peers or []
        sent_count = 0
        
        if self.node and hasattr(self.node, 'peers'):
            for peer_id in self.node.peers:
                if peer_id not in exclude_peers:
                    if await self.send_message(peer_id, message):
                        sent_count += 1
        
        return sent_count
    
    async def get_node_status(self) -> Dict[str, Any]:
        """Get comprehensive node status."""
        status = {
            'node_id': str(self.node.node_id) if self.node else 'unknown',
            'node_name': getattr(self.config, 'node_name', 'unknown'),
            'node_type': getattr(self.config, 'node_type', 'unknown'),
            'uptime_seconds': self.metrics['uptime_seconds'],
            'running': self.running,
            'timestamp': datetime.now().isoformat(),
            'metrics': self.metrics.copy(),
            'network': {
                'peers_connected': self.metrics['peers_connected'],
                'messages_sent': self.metrics['messages_sent'],
                'messages_received': self.metrics['messages_received'],
                'bytes_sent': self.metrics['bytes_sent'],
                'bytes_received': self.metrics['bytes_received']
            },
            'system': {
                'cpu_usage': self.metrics['cpu_usage'],
                'memory_usage': self.metrics['memory_usage'],
                'platform': sys.platform,
                'python_version': sys.version
            }
        }
        
        return status
    
    async def print_comprehensive_status(self):
        """Print comprehensive node status."""
        status = await self.get_node_status()
        
        self.logger.info("📊 === Enhanced CSP Network Status ===")
        self.logger.info(f"🆔 Node ID: {status['node_id'][:16]}...")
        self.logger.info(f"🏷️  Name: {status['node_name']} ({status['node_type']})")
        self.logger.info(f"⏱️  Uptime: {format_duration(status['uptime_seconds'])}")
        
        # Network status
        net = status['network']
        self.logger.info(f"🌐 Network: {net['peers_connected']} peers")
        self.logger.info(f"📨 Messages: {net['messages_sent']} sent, {net['messages_received']} received")
        self.logger.info(f"📊 Traffic: {format_bytes(net['bytes_sent'])} sent, {format_bytes(net['bytes_received'])} received")
        
        # System status
        sys_info = status['system']
        self.logger.info(f"🔧 System: {sys_info['cpu_usage']:.1f}% CPU, {format_bytes(sys_info['memory_usage'] * 1024 * 1024)} RAM")
        
        self.logger.info("=" * 45)
    
    async def start_dashboard(self):
        """Start the web dashboard."""
        try:
            # Simple HTTP status server
            from http.server import HTTPServer, BaseHTTPRequestHandler
            import threading
            import json
            
            class StatusHandler(BaseHTTPRequestHandler):
                def do_GET(self):
                    if self.path == '/status':
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.send_header('Access-Control-Allow-Origin', '*')
                        self.end_headers()
                        
                        # Get status from main instance
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        status = loop.run_until_complete(enhanced_csp_main.get_node_status())
                        loop.close()
                        
                        self.wfile.write(json.dumps(status, indent=2).encode())
                    else:
                        self.send_response(200)
                        self.send_header('Content-type', 'text/html')
                        self.end_headers()
                        
                        html = """
                        <!DOCTYPE html>
                        <html>
                        <head><title>Enhanced CSP Network Dashboard</title></head>
                        <body>
                            <h1>Enhanced CSP Network Node</h1>
                            <p><a href="/status">JSON Status</a></p>
                            <div id="status"></div>
                            <script>
                                setInterval(() => {
                                    fetch('/status')
                                        .then(r => r.json())
                                        .then(data => {
                                            document.getElementById('status').innerHTML = 
                                                '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
                                        });
                                }, 5000);
                            </script>
                        </body>
                        </html>
                        """
                        self.wfile.write(html.encode())
                
                def log_message(self, format, *args):
                    pass  # Suppress HTTP logs
            
            # Start server in background thread
            def start_server():
                server = HTTPServer(('0.0.0.0', self.dashboard_port), StatusHandler)
                server.serve_forever()
            
            thread = threading.Thread(target=start_server, daemon=True)
            thread.start()
            
            self.dashboard_enabled = True
            self.logger.info(f"🌐 Dashboard started at http://localhost:{self.dashboard_port}")
            
        except Exception as e:
            self.logger.error(f"❌ Dashboard start failed: {e}")
    
    async def run_benchmark(self):
        """Run comprehensive network benchmark."""
        self.logger.info("📊 Starting network benchmark...")
        
        try:
            from enhanced_csp.network.benchmark import NetworkBenchmark
            benchmark = NetworkBenchmark()
            report = await benchmark.run_all_benchmarks(mode='quick')
            
            self.logger.info("🏆 Benchmark Results:")
            self.logger.info(f"   🚀 Peak Throughput: {report['summary']['max_throughput']:.0f} msg/s")
            self.logger.info(f"   ⚡ Avg Latency: {report['summary']['avg_latency_ms']:.2f} ms")
            self.logger.info(f"   🎯 Grade: {report['performance_analysis']['performance_grade']}")
            
        except ImportError:
            self.logger.warning("⚠️  Benchmark module not available")
        except Exception as e:
            self.logger.error(f"❌ Benchmark failed: {e}")
    
    async def graceful_shutdown(self):
        """Perform graceful shutdown of all components."""
        self.logger.info("🛑 Initiating graceful shutdown...")
        
        self.running = False
        self.shutdown_event.set()
        
        try:
            # Stop components in reverse order
            if self.intelligent_batching:
                await self.intelligent_batching.stop()
            
            if self.adaptive_compression:
                await self.adaptive_compression.stop()
            
            if self.performance_optimizer:
                await self.performance_optimizer.stop()
            
            if self.security_orchestrator:
                await self.security_orchestrator.shutdown()
            
            # Stop node last
            if self.node:
                await self.node.stop()
            
            self.logger.info("✅ Graceful shutdown completed")
            
        except Exception as e:
            self.logger.error(f"❌ Shutdown error: {e}")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"🛑 Received signal {signum}, initiating shutdown...")
            asyncio.create_task(self.graceful_shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main entry point with comprehensive argument parsing."""
    parser = argparse.ArgumentParser(
        description="Enhanced CSP Network - Full-Featured Node",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --genesis                    # Start as genesis node
  python main.py --connect genesis.peoplesainetwork.com  # Connect to network
  python main.py --benchmark                  # Run performance tests
  python main.py --dashboard --port 30301     # Start with web dashboard
        """
    )
    
    # === Operation Mode ===
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--genesis", action="store_true",
                           help="Start as genesis node")
    mode_group.add_argument("--connect", metavar="HOST", 
                           help="Connect to genesis host")
    mode_group.add_argument("--benchmark", action="store_true",
                           help="Run network benchmark")
    
    # === Network Configuration ===
    net_group = parser.add_argument_group("Network Configuration")
    net_group.add_argument("--genesis-host", default="genesis.peoplesainetwork.com",
                          help="Genesis node hostname")
    net_group.add_argument("--genesis-port", type=int, default=30300,
                          help="Genesis node port")
    net_group.add_argument("--port", type=int, default=30301,
                          help="Local listen port")
    net_group.add_argument("--listen-address", default="0.0.0.0",
                          help="Listen address")
    net_group.add_argument("--max-connections", type=int, default=100,
                          help="Maximum peer connections")
    net_group.add_argument("--connection-timeout", type=int, default=30,
                          help="Connection timeout (seconds)")
    net_group.add_argument("--max-message-size", type=int, default=10485760,
                          help="Maximum message size (bytes)")
    
    # === Node Configuration ===
    node_group = parser.add_argument_group("Node Configuration")
    node_group.add_argument("--node-name", default="enhanced-csp-node",
                           help="Node name")
    node_group.add_argument("--data-dir", default="./network_data",
                           help="Data directory")
    node_group.add_argument("--max-peers", type=int, default=50,
                           help="Maximum peers in mesh")
    node_group.add_argument("--max-storage", type=int, default=1000,
                           help="Maximum storage (MB)")
    node_group.add_argument("--max-compute", type=int, default=100,
                           help="Maximum compute units")
    
    # === Protocol Features ===
    proto_group = parser.add_argument_group("Protocol Features")
    proto_group.add_argument("--enable-mdns", action="store_true", default=True,
                            help="Enable mDNS discovery")
    proto_group.add_argument("--enable-upnp", action="store_true", default=True,
                            help="Enable UPnP port mapping")
    proto_group.add_argument("--enable-nat", action="store_true", default=True,
                            help="Enable NAT traversal")
    proto_group.add_argument("--enable-quic", action="store_true", default=True,
                            help="Enable QUIC protocol")
    proto_group.add_argument("--enable-dht", action="store_true", default=True,
                            help="Enable DHT")
    proto_group.add_argument("--enable-mesh", action="store_true", default=True,
                            help="Enable mesh networking")
    proto_group.add_argument("--enable-dns-overlay", action="store_true", default=True,
                            help="Enable DNS overlay")
    
    # === Mesh Configuration ===
    mesh_group = parser.add_argument_group("Mesh Configuration")
    mesh_group.add_argument("--mesh-topology", default="adaptive_partial",
                           choices=["full", "partial", "adaptive_partial"],
                           help="Mesh topology type")
    mesh_group.add_argument("--enable-super-peers", action="store_true", default=True,
                           help="Enable super peer mode")
    mesh_group.add_argument("--routing-interval", type=int, default=10,
                           help="Routing update interval (seconds)")
    mesh_group.add_argument("--link-quality-threshold", type=float, default=0.5,
                           help="Link quality threshold")
    mesh_group.add_argument("--enable-multi-hop", action="store_true", default=True,
                           help="Enable multi-hop routing")
    mesh_group.add_argument("--max-hops", type=int, default=10,
                           help="Maximum hop count")
    
    # === Security Configuration ===
    sec_group = parser.add_argument_group("Security Configuration")
    sec_group.add_argument("--enable-encryption", action="store_true", default=True,
                          help="Enable message encryption")
    sec_group.add_argument("--enable-auth", action="store_true", default=True,
                          help="Enable peer authentication")
    sec_group.add_argument("--enable-tls", action="store_true", default=True,
                          help="Enable TLS transport")
    sec_group.add_argument("--key-size", type=int, default=2048,
                          help="Encryption key size")
    sec_group.add_argument("--cipher-suite", default="AES-256-GCM",
                          help="Cipher suite")
    
    # === DNS Configuration ===
    dns_group = parser.add_argument_group("DNS Configuration")
    dns_group.add_argument("--enable-dnssec", action="store_true", default=True,
                          help="Enable DNSSEC")
    dns_group.add_argument("--dns-root-domain", default=".csp",
                          help="DNS root domain")
    dns_group.add_argument("--dns-ttl", type=int, default=3600,
                          help="DNS TTL (seconds)")
    dns_group.add_argument("--dns-cache-size", type=int, default=10000,
                          help="DNS cache size")
    dns_group.add_argument("--dns-seed", 
                          help="DNS seed domain")
    
    # === Routing Configuration ===
    route_group = parser.add_argument_group("Routing Configuration")
    route_group.add_argument("--enable-multipath", action="store_true", default=True,
                            help="Enable multipath routing")
    route_group.add_argument("--enable-ml-routing", action="store_true", default=True,
                            help="Enable ML-based routing")
    route_group.add_argument("--enable-adaptive-routing", action="store_true", default=True,
                            help="Enable adaptive routing")
    route_group.add_argument("--max-paths", type=int, default=3,
                            help="Max paths per destination")
    route_group.add_argument("--failover-threshold", type=int, default=500,
                            help="Failover threshold (ms)")
    route_group.add_argument("--enable-congestion-control", action="store_true", default=True,
                            help="Enable congestion control")
    route_group.add_argument("--enable-qos", action="store_true", default=True,
                            help="Enable QoS")
    
    # === Node Capabilities ===
    cap_group = parser.add_argument_group("Node Capabilities")
    cap_group.add_argument("--enable-relay", action="store_true", default=True,
                          help="Enable message relay")
    cap_group.add_argument("--enable-storage", action="store_true", default=True,
                          help="Enable data storage")
    cap_group.add_argument("--enable-compute", action="store_true", default=True,
                          help="Enable computation")
    
    # === Performance and Optimizations ===
    perf_group = parser.add_argument_group("Performance and Optimizations")
    perf_group.add_argument("--enable-optimizations", action="store_true", default=True,
                           help="Enable performance optimizations")
    perf_group.add_argument("--enable-compression", action="store_true", default=True,
                           help="Enable adaptive compression")
    perf_group.add_argument("--enable-batching", action="store_true", default=True,
                           help="Enable intelligent batching")
    perf_group.add_argument("--enable-zero-copy", action="store_true", default=True,
                           help="Enable zero-copy I/O")
    
    # === Monitoring and Debugging ===
    monitor_group = parser.add_argument_group("Monitoring and Debugging")
    monitor_group.add_argument("--enable-dashboard", action="store_true", default=False,
                              help="Enable web dashboard")
    monitor_group.add_argument("--dashboard-port", type=int, default=8080,
                              help="Dashboard port")
    monitor_group.add_argument("--log-level", default="INFO",
                              choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                              help="Logging level")
    monitor_group.add_argument("--log-file",
                              help="Log file path")
    monitor_group.add_argument("--metrics-interval", type=int, default=60,
                              help="Metrics collection interval")
    
    # === Bootstrap and Discovery ===
    bootstrap_group = parser.add_argument_group("Bootstrap and Discovery")
    bootstrap_group.add_argument("--bootstrap-nodes", nargs="+",
                                help="Additional bootstrap nodes (multiaddr format)")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            *([logging.FileHandler(args.log_file)] if args.log_file else [])
        ]
    )
    
    # Handle connect shorthand
    if args.connect:
        args.genesis_host = args.connect
        args.genesis = False
    
    # Print startup banner
    print("🚀 Enhanced CSP Network - Full-Featured Node")
    print("=" * 60)
    print(f"🌐 Mode: {'Genesis' if args.genesis else 'Client'}")
    print(f"🔌 Port: {args.port}")
    if not args.genesis:
        print(f"🔗 Genesis: {args.genesis_host}:{args.genesis_port}")
    print(f"🏷️  Node: {args.node_name}")
    print(f"📊 Features: All enabled" if args.enable_optimizations else "📊 Features: Basic")
    if args.enable_dashboard:
        print(f"🌐 Dashboard: http://localhost:{args.dashboard_port}")
    print()
    
    # Global reference for dashboard
    global enhanced_csp_main
    enhanced_csp_main = EnhancedCSPMain()
    enhanced_csp_main.dashboard_port = args.dashboard_port
    
    try:
        # Setup signal handlers
        enhanced_csp_main.setup_signal_handlers()
        
        # Special mode: benchmark
        if args.benchmark:
            await enhanced_csp_main.run_benchmark()
            return 0
        
        # Create configuration
        config = enhanced_csp_main.create_comprehensive_config(args)
        enhanced_csp_main.config = config
        
        # Start the network node
        if await enhanced_csp_main.start_node():
            print("🎉 Enhanced CSP Network node started successfully!")
            
            # Connect to network (unless genesis)
            if not args.genesis:
                print("🔍 Connecting to Enhanced CSP Network...")
                await enhanced_csp_main.connect_to_network()
            
            print("🔄 Node operational! Press Ctrl+C to stop")
            print("=" * 60)
            
            # Run main event loop
            await enhanced_csp_main.run_main_event_loop()
        else:
            print("❌ Failed to start Enhanced CSP Network node")
            return 1
            
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        traceback.print_exc()
        return 1
    finally:
        if enhanced_csp_main.running:
            await enhanced_csp_main.graceful_shutdown()
        print("👋 Enhanced CSP Network node shutdown complete")
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

#!/usr/bin/env python3
"""
Enhanced CSP Network - Genesis Connection Handler
This script ensures proper connection to the genesis server and handles all networking.
"""

import asyncio
import sys
import signal
import logging
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import time
import socket
import traceback

# Add project root to Python path
current_file = Path(__file__).resolve()
project_root = current_file.parent
sys.path.insert(0, str(project_root))

try:
    from enhanced_csp.network.core.config import NetworkConfig, P2PConfig, MeshConfig, SecurityConfig
    from enhanced_csp.network.core.types import NodeID, NodeCapabilities
    from enhanced_csp.network.core.node import NetworkNode, EnhancedCSPNetwork
    from enhanced_csp.network.utils import get_logger, setup_logging
    from enhanced_csp.network import create_network, create_node
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Enhanced CSP imports failed: {e}")
    print("Make sure you're running from the project root directory")
    print("Try: pip install -r requirements-lock.txt")
    
    # Try to provide more helpful error information
    import sys
    from pathlib import Path
    
    print(f"🔍 Current directory: {Path.cwd()}")
    print(f"🐍 Python path: {sys.path[:3]}...")
    
    # Check if the enhanced_csp directory exists
    enhanced_csp_path = Path("enhanced_csp")
    if enhanced_csp_path.exists():
        print(f"✅ enhanced_csp directory found")
        network_path = enhanced_csp_path / "network"
        if network_path.exists():
            print(f"✅ enhanced_csp/network directory found")
            utils_path = network_path / "utils"
            if utils_path.exists():
                print(f"✅ enhanced_csp/network/utils directory found")
                init_file = utils_path / "__init__.py"
                if init_file.exists():
                    print(f"✅ utils/__init__.py exists")
                else:
                    print(f"❌ utils/__init__.py missing")
            else:
                print(f"❌ enhanced_csp/network/utils directory missing")
        else:
            print(f"❌ enhanced_csp/network directory missing")
    else:
        print(f"❌ enhanced_csp directory missing")
    
    IMPORTS_AVAILABLE = False
    sys.exit(1)

# Global logger
logger = None




class GenesisNetworkConnector:
    """Handles connection to genesis server and network management."""
    
    def __init__(self):
        self.network: Optional[EnhancedCSPNetwork] = None
        self.node: Optional[NetworkNode] = None
        self.running = False
        self.config: Optional[NetworkConfig] = None
        self.connection_attempts = 0
        self.max_connection_attempts = 50
        self.start_time = time.time()
        
    def setup_logging(self, level: str = "INFO"):
        """Setup comprehensive logging."""
        global logger
        setup_logging(level=level)
        logger = get_logger("genesis_connector")
        
        # Check if logger is the expected type before adding handlers
        if hasattr(logger, 'addHandler'):
            # Additional console handler for user feedback
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            formatter = logging.Formatter('%(asctime)s | %(levelname)8s | %(message)s')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        else:
            # If it's a StructuredAdapter or other type, just use basic logging
            logging.basicConfig(
                level=getattr(logging, level.upper()),
                format='%(asctime)s | %(levelname)8s | %(message)s',
                handlers=[logging.StreamHandler(sys.stdout)]
            )
            logger = logging.getLogger("genesis_connector")
        
    def create_genesis_config(self, 
                             genesis_host: str = "genesis.peoplesainetwork.com",
                             genesis_port: int = 30300,
                             local_port: int = 30301,
                             node_name: str = "csp-node",
                             enable_all_features: bool = True) -> NetworkConfig:
        """Create optimized configuration for genesis connection."""
        
        logger.info(f"🔧 Creating network configuration...")
        logger.info(f"📡 Genesis server: {genesis_host}:{genesis_port}")
        logger.info(f"🌐 Local port: {local_port}")
        
        # Create main configuration
        config = NetworkConfig()
        
        # P2P Configuration - optimized for genesis connection
        config.p2p = P2PConfig()
        config.p2p.listen_address = "0.0.0.0"
        config.p2p.listen_port = local_port
        config.p2p.enable_mdns = True  # Local network discovery
        config.p2p.enable_upnp = True  # NAT traversal
        config.p2p.enable_nat_traversal = True
        config.p2p.connection_timeout = 30
        config.p2p.max_connections = 100
        
        # Genesis bootstrap configuration
        genesis_multiaddr = f"/ip4/{genesis_host}/tcp/{genesis_port}"
        config.p2p.bootstrap_nodes = [genesis_multiaddr]
        config.p2p.dns_seed_domain = "peoplesainetwork.com"
        
        # Fallback genesis servers
        config.p2p.bootstrap_nodes.extend([
            f"/ip4/147.75.77.187/tcp/{genesis_port}",  # Backup genesis
            f"/dns4/seed1.peoplesainetwork.com/tcp/{genesis_port}",
            f"/dns4/seed2.peoplesainetwork.com/tcp/{genesis_port}",
            f"/dns4/bootstrap.peoplesainetwork.com/tcp/{genesis_port}"
        ])
        
        # Mesh networking configuration
        config.mesh = MeshConfig()
        config.mesh.enable_super_peers = True
        config.mesh.max_peers = 50
        config.mesh.topology_type = "dynamic_partial"
        config.mesh.heartbeat_interval = 30
        config.mesh.redundancy_factor = 3
        
        # Security configuration
        config.security = SecurityConfig()
        config.security.enable_encryption = True
        config.security.enable_authentication = True
        config.security.key_size = 2048
        
        # Node identity and capabilities
        config.node_name = node_name
        config.node_type = "standard"
        config.network_id = "enhanced-csp-mainnet"
        config.protocol_version = "1.0.0"
        
        # Node capabilities
        capabilities = NodeCapabilities()
        capabilities.relay = True
        capabilities.storage = False  # Start basic, upgrade later
        capabilities.compute = False
        capabilities.quantum = enable_all_features
        config.capabilities = capabilities
        
        # Advanced features
        config.enable_dht = True
        config.enable_mesh = True
        config.enable_dns = True
        config.enable_adaptive_routing = True
        config.enable_metrics = True
        config.enable_ipv6 = True
        
        # Performance tuning
        config.max_message_size = 1024 * 1024  # 1MB
        config.enable_compression = True
        config.gossip_interval = 5
        config.gossip_fanout = 6
        config.metrics_interval = 60
        
        # Data directory
        config.data_dir = Path("./network_data")
        config.data_dir.mkdir(exist_ok=True)
        
        # Connection retry configuration
        config.bootstrap_retry_delay = 5
        config.max_bootstrap_attempts = 10
        
        logger.info(f"✅ Configuration created successfully")
        return config
    
    async def check_genesis_connectivity(self, host: str, port: int) -> bool:
        """Check if genesis server is reachable."""
        logger.info(f"🔍 Testing connectivity to {host}:{port}...")
        
        try:
            # TCP connectivity test
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=10
            )
            writer.close()
            await writer.wait_closed()
            logger.info(f"✅ Genesis server is reachable")
            return True
        except asyncio.TimeoutError:
            logger.warning(f"⚠️  Connection timeout to {host}:{port}")
            return False
        except Exception as e:
            logger.warning(f"⚠️  Connection failed: {e}")
            return False
    
    async def start_network(self, config: NetworkConfig) -> bool:
        """Start the Enhanced CSP Network."""
        try:
            self.config = config
            logger.info("🚀 Starting Enhanced CSP Network...")
            
            # Create the network instance
            self.network = create_network(config)
            
            # Start the network
            success = await self.network.start()
            if not success:
                logger.error("❌ Failed to start network")
                return False
                
            self.running = True
            logger.info("✅ Enhanced CSP Network started successfully!")
            logger.info(f"🆔 Network ID: {self.network.node_id}")
            
            # Get the default node
            self.node = self.network.get_node("default")
            if self.node:
                logger.info(f"🏷️  Node ID: {self.node.node_id}")
                logger.info(f"📊 Capabilities: {self.node.capabilities}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Network startup failed: {e}")
            logger.debug(traceback.format_exc())
            return False
    
    async def connect_to_genesis(self) -> bool:
        """Attempt to connect to the genesis server with retry logic."""
        if not self.network or not self.node:
            logger.error("❌ Network not started - cannot connect to genesis")
            return False
        
        logger.info("🔍 Attempting to connect to genesis network...")
        
        max_attempts = self.max_connection_attempts
        attempt = 0
        
        while attempt < max_attempts and self.running:
            try:
                self.connection_attempts += 1
                attempt += 1
                
                logger.info(f"🔄 Connection attempt {attempt}/{max_attempts}")
                
                # Check if we have any peers
                if hasattr(self.node, 'peers') and len(self.node.peers) > 0:
                    logger.info(f"🎉 Connected to {len(self.node.peers)} peers!")
                    self._log_connected_peers()
                    return True
                
                # Check discovery system
                if hasattr(self.node, 'discovery') and self.node.discovery:
                    try:
                        peers = await self.node.discovery.find_peers(count=5)
                        if peers:
                            logger.info(f"🎉 Discovered {len(peers)} peers via discovery!")
                            for peer in peers[:3]:  # Log first 3
                                peer_id = peer.get('node_id', 'unknown')
                                source = peer.get('source', 'unknown')
                                logger.info(f"   📡 Peer: {peer_id[:16]}... (via {source})")
                            return True
                    except Exception as e:
                        logger.debug(f"Discovery attempt failed: {e}")
                
                # Check DHT for peers
                if hasattr(self.node, 'dht') and self.node.dht:
                    try:
                        # Try to find peers through DHT
                        if hasattr(self.node.dht, 'get_peers'):
                            dht_peers = await self.node.dht.get_peers()
                            if dht_peers:
                                logger.info(f"🎉 Found {len(dht_peers)} peers via DHT!")
                                return True
                    except Exception as e:
                        logger.debug(f"DHT lookup failed: {e}")
                
                # Log progress every 5 attempts
                if attempt % 5 == 0:
                    logger.info(f"🔄 Still searching for peers... (attempt {attempt}/{max_attempts})")
                    logger.info(f"💡 Network has been running for {time.time() - self.start_time:.1f} seconds")
                
                # Wait before next attempt
                await asyncio.sleep(3)
                
            except Exception as e:
                logger.debug(f"Connection attempt {attempt} failed: {e}")
                await asyncio.sleep(2)
        
        logger.warning("⚠️  Could not connect to genesis network within timeout")
        logger.info("💡 Node will continue running and may connect later")
        logger.info(f"📊 Total connection attempts: {self.connection_attempts}")
        return False
    
    def _log_connected_peers(self):
        """Log information about connected peers."""
        if not self.node or not hasattr(self.node, 'peers'):
            return
            
        for peer_id, peer_info in list(self.node.peers.items())[:5]:  # Log first 5
            logger.info(f"   🤝 Peer: {peer_id}")
    
    async def run_network_loop(self):
        """Main network event loop."""
        logger.info("🔄 Starting network event loop...")
        
        # Network status monitoring
        last_status_log = time.time()
        status_interval = 120  # Log status every 2 minutes
        
        while self.running:
            try:
                current_time = time.time()
                
                # Periodic status logging
                if current_time - last_status_log >= status_interval:
                    await self._log_network_status()
                    last_status_log = current_time
                
                # Check if network is still healthy
                if self.network and not self.network.is_running:
                    logger.warning("⚠️  Network appears to have stopped")
                    break
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"❌ Network loop error: {e}")
                await asyncio.sleep(5)
    
    async def _log_network_status(self):
        """Log current network status."""
        if not self.network or not self.node:
            return
            
        uptime = time.time() - self.start_time
        peer_count = len(self.node.peers) if hasattr(self.node, 'peers') else 0
        
        logger.info(f"📊 Network Status:")
        logger.info(f"   ⏱️  Uptime: {uptime:.1f} seconds")
        logger.info(f"   🤝 Connected peers: {peer_count}")
        logger.info(f"   🔗 Connection attempts: {self.connection_attempts}")
        
        # Log metrics if available
        if hasattr(self.node, 'get_stats'):
            try:
                stats = self.node.get_stats()
                logger.info(f"   📈 Messages sent: {stats.get('messages_sent', 0)}")
                logger.info(f"   📥 Messages received: {stats.get('messages_received', 0)}")
            except Exception:
                pass
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"🛑 Received signal {signum}, initiating shutdown...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def shutdown(self):
        """Graceful shutdown of the network."""
        logger.info("🛑 Shutting down Enhanced CSP Network...")
        
        self.running = False
        
        try:
            if self.network:
                await self.network.stop()
                logger.info("✅ Network stopped successfully")
        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Enhanced CSP Network - Genesis Connector")
    
    # Connection settings
    parser.add_argument("--genesis-host", default="genesis.peoplesainetwork.com",
                       help="Genesis node hostname")
    parser.add_argument("--genesis-port", type=int, default=30300,
                       help="Genesis node port")
    parser.add_argument("--local-port", type=int, default=30301,
                       help="Local node listen port")
    
    # Node settings
    parser.add_argument("--node-name", default="csp-node",
                       help="Name for this node")
    parser.add_argument("--data-dir", type=Path, default=Path("./network_data"),
                       help="Data directory for node storage")
    
    # Feature flags
    parser.add_argument("--disable-features", action="store_true",
                       help="Disable advanced features for basic connection")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level")
    
    # Connection options
    parser.add_argument("--max-connection-attempts", type=int, default=50,
                       help="Maximum connection attempts to genesis")
    parser.add_argument("--quick-start", action="store_true",
                       help="Skip connectivity checks and start immediately")
    
    args = parser.parse_args()
    
    # Create connector
    connector = GenesisNetworkConnector()
    connector.setup_logging(args.log_level)
    connector.setup_signal_handlers()
    
    print("🚀 Enhanced CSP Network - Genesis Connector")
    print("=" * 60)
    print(f"🌐 Genesis: {args.genesis_host}:{args.genesis_port}")
    print(f"🔌 Local Port: {args.local_port}")
    print(f"🏷️  Node Name: {args.node_name}")
    print(f"📊 Log Level: {args.log_level}")
    print(f"🔄 Max Connection Attempts: {args.max_connection_attempts}")
    print()
    
    try:
        # Set connection attempt limit
        connector.max_connection_attempts = args.max_connection_attempts
        
        # Pre-flight connectivity check
        if not args.quick_start:
            connectivity_ok = await connector.check_genesis_connectivity(
                args.genesis_host, args.genesis_port
            )
            if not connectivity_ok:
                logger.warning("⚠️  Genesis server not reachable, but continuing anyway...")
                logger.info("💡 Use --quick-start to skip connectivity checks")
        
        # Create configuration
        config = connector.create_genesis_config(
            genesis_host=args.genesis_host,
            genesis_port=args.genesis_port,
            local_port=args.local_port,
            node_name=args.node_name,
            enable_all_features=not args.disable_features
        )
        
        # Set data directory
        config.data_dir = args.data_dir
        config.data_dir.mkdir(exist_ok=True)
        
        # Start the network
        if await connector.start_network(config):
            logger.info("🎉 Network started successfully!")
            
            # Attempt genesis connection
            logger.info("🔗 Attempting to connect to genesis network...")
            genesis_connected = await connector.connect_to_genesis()
            
            if genesis_connected:
                logger.info("🌟 Successfully connected to genesis network!")
            else:
                logger.info("💡 Continuing without genesis connection - may connect later")
            
            # Run the main network loop
            await connector.run_network_loop()
            
        else:
            logger.error("❌ Failed to start network")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("🛑 Received interrupt signal")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)
    finally:
        await connector.shutdown()
        logger.info("👋 Enhanced CSP Network shutdown complete")


if __name__ == "__main__":
    if not IMPORTS_AVAILABLE:
        sys.exit(1)
    
    asyncio.run(main())
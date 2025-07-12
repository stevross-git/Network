#!/usr/bin/env python3
"""
Enhanced CSP Network - Node Runner
Connects to the Peoples AI Network genesis node and starts the network stack.
"""

import asyncio
import sys
import signal
import logging
import argparse
from pathlib import Path
from typing import Optional, List
import json
import time

# Add project root to Python path
current_file = Path(__file__).resolve()
project_root = current_file.parent
sys.path.insert(0, str(project_root))

try:
    from enhanced_csp.network.core.config import NetworkConfig, P2PConfig, MeshConfig, SecurityConfig
    from enhanced_csp.network.core.types import NodeID
    from enhanced_csp.network.core.node import NetworkNode
    from enhanced_csp.network.utils import get_logger, setup_logging
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Enhanced CSP imports failed: {e}")
    print("Make sure you're running from the project root directory")
    IMPORTS_AVAILABLE = False
    sys.exit(1)

# Setup logging
setup_logging(level="INFO")
logger = get_logger("network_runner")


class NetworkRunner:
    """Enhanced CSP Network node runner with genesis connection."""
    
    def __init__(self):
        self.node: Optional[NetworkNode] = None
        self.running = False
        self.config: Optional[NetworkConfig] = None
        
    def create_config(self, 
                     genesis_host: str = "genesis.peoplesainetwork.com",
                     genesis_port: int = 4001,
                     local_port: int = 4002,
                     node_name: str = "csp-node",
                     enable_dashboard: bool = True,
                     enable_optimizations: bool = True) -> NetworkConfig:
        """Create network configuration for connecting to genesis."""
        
        # Create the main network configuration
        config = NetworkConfig()
        
        # Configure P2P settings
        config.p2p = P2PConfig()
        config.p2p.listen_address = "0.0.0.0"
        config.p2p.listen_port = local_port
        config.p2p.enable_mdns = True
        config.p2p.enable_upnp = True
        config.p2p.enable_nat_traversal = True
        
        # Add genesis node as bootstrap
        genesis_multiaddr = f"/ip4/{genesis_host}/tcp/{genesis_port}"
        config.p2p.bootstrap_nodes = [genesis_multiaddr]
        
        # Optional: Add DNS resolution for the genesis host
        config.p2p.dns_seed_domain = "peoplesainetwork.com"
        
        # Configure mesh networking
        config.mesh = MeshConfig()
        config.mesh.enable_super_peers = True
        config.mesh.max_peers = 50
        config.mesh.topology_type = "adaptive_partial"
        
        # Security configuration
        config.security = SecurityConfig()
        config.security.enable_encryption = True
        config.security.enable_authentication = True
        
        # Node identity
        config.node_name = node_name
        config.node_type = "standard"
        
        # Enable advanced features
        config.enable_dht = True
        config.enable_mesh = True
        config.enable_dns = True
        config.enable_adaptive_routing = enable_optimizations
        
        # Data directory
        config.data_dir = Path("./network_data")
        config.data_dir.mkdir(exist_ok=True)
        
        logger.info(f"🔧 Configuration created for genesis: {genesis_host}:{genesis_port}")
        logger.info(f"🌐 Local node will listen on port: {local_port}")
        
        return config
    
    async def start_node(self, config: NetworkConfig) -> bool:
        """Start the network node with the given configuration."""
        try:
            self.config = config
            
            logger.info("🚀 Starting Enhanced CSP Network Node...")
            logger.info(f"🏷️  Node name: {config.node_name}")
            logger.info(f"🌐 Listen address: {config.p2p.listen_address}:{config.p2p.listen_port}")
            logger.info(f"🔗 Genesis nodes: {config.p2p.bootstrap_nodes}")
            
            # Create and start the network node
            self.node = NetworkNode(config)
            await self.node.start()
            
            self.running = True
            logger.info("✅ Network node started successfully!")
            
            # Log node information
            logger.info(f"🆔 Node ID: {self.node.node_id}")
            logger.info(f"📊 Capabilities: {self.node.capabilities}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start network node: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def connect_to_genesis(self) -> bool:
        """Attempt to connect to the genesis node."""
        if not self.node:
            logger.error("❌ Node not started - cannot connect to genesis")
            return False
        
        try:
            logger.info("🔍 Attempting to connect to genesis node...")
            
            # Wait for discovery to find peers
            max_attempts = 30
            attempt = 0
            
            while attempt < max_attempts and self.running:
                try:
                    # Check if we have discovered any peers
                    if hasattr(self.node, 'discovery') and self.node.discovery:
                        peers = await self.node.discovery.find_peers(count=5)
                        if peers:
                            logger.info(f"🎉 Discovered {len(peers)} peers!")
                            for peer in peers:
                                peer_id = peer.get('node_id', 'unknown')
                                source = peer.get('source', 'unknown')
                                logger.info(f"   📡 Peer: {peer_id[:16]}... (via {source})")
                            return True
                    
                    # Check connected peers through node
                    if hasattr(self.node, 'peers') and len(self.node.peers) > 0:
                        logger.info(f"🌐 Connected to {len(self.node.peers)} peers!")
                        return True
                    
                except Exception as e:
                    logger.debug(f"Discovery attempt {attempt + 1} failed: {e}")
                
                attempt += 1
                await asyncio.sleep(2)  # Wait 2 seconds between attempts
                
                if attempt % 5 == 0:
                    logger.info(f"🔄 Still searching for peers... (attempt {attempt}/{max_attempts})")
            
            logger.warning("⚠️  Could not connect to genesis node within timeout")
            logger.info("💡 Node will continue running and may connect later")
            return False
            
        except Exception as e:
            logger.error(f"❌ Genesis connection failed: {e}")
            return False
    
    async def run_network_loop(self):
        """Main network event loop."""
        logger.info("🔄 Starting network event loop...")
        
        try:
            # Attempt genesis connection
            await self.connect_to_genesis()
            
            # Main running loop
            while self.running:
                try:
                    # Periodic status updates
                    await self._print_status()
                    
                    # Process network events
                    if self.node:
                        # TODO: Process incoming messages, maintain connections, etc.
                        pass
                    
                    await asyncio.sleep(30)  # Status update every 30 seconds
                    
                except Exception as e:
                    logger.error(f"❌ Error in network loop: {e}")
                    await asyncio.sleep(5)
            
        except asyncio.CancelledError:
            logger.info("🛑 Network loop cancelled")
        except Exception as e:
            logger.error(f"❌ Network loop failed: {e}")
        finally:
            await self.stop_node()
    
    async def _print_status(self):
        """Print current network status."""
        if not self.node:
            return
        
        try:
            peer_count = len(getattr(self.node, 'peers', {}))
            uptime = time.time() - getattr(self.node, 'start_time', time.time())
            
            logger.info(f"📊 Network Status:")
            logger.info(f"   🆔 Node ID: {self.node.node_id}")
            logger.info(f"   🌐 Connected Peers: {peer_count}")
            logger.info(f"   ⏱️  Uptime: {uptime:.1f}s")
            
            # Additional status from components
            if hasattr(self.node, 'discovery') and self.node.discovery:
                discovered = len(getattr(self.node.discovery, 'discovered_peers', {}))
                logger.info(f"   🔍 Discovered Peers: {discovered}")
            
            if hasattr(self.node, 'metrics'):
                # Log basic metrics if available
                pass
                
        except Exception as e:
            logger.debug(f"Status update failed: {e}")
    
    async def stop_node(self):
        """Stop the network node gracefully."""
        if not self.node:
            return
        
        try:
            logger.info("🛑 Stopping network node...")
            self.running = False
            
            await self.node.stop()
            logger.info("✅ Network node stopped successfully")
            
        except Exception as e:
            logger.error(f"❌ Error stopping node: {e}")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"🛑 Received signal {signum}, initiating shutdown...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main entry point for the network runner."""
    parser = argparse.ArgumentParser(description="Enhanced CSP Network Node Runner")
    
    # Connection settings
    parser.add_argument("--genesis-host", default="genesis.peoplesainetwork.com",
                       help="Genesis node hostname")
    parser.add_argument("--genesis-port", type=int, default=4001,
                       help="Genesis node port")
    parser.add_argument("--local-port", type=int, default=4002,
                       help="Local node listen port")
    
    # Node settings
    parser.add_argument("--node-name", default="csp-node",
                       help="Name for this node")
    parser.add_argument("--data-dir", type=Path, default=Path("./network_data"),
                       help="Data directory for node storage")
    
    # Feature flags
    parser.add_argument("--disable-optimizations", action="store_true",
                       help="Disable performance optimizations")
    parser.add_argument("--disable-dashboard", action="store_true",
                       help="Disable web dashboard")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level")
    
    # Network options
    parser.add_argument("--bootstrap-nodes", nargs="+",
                       help="Additional bootstrap nodes (multiaddr format)")
    
    args = parser.parse_args()
    
    # Setup logging with specified level
    setup_logging(level=args.log_level)
    
    print("🚀 Enhanced CSP Network - Node Runner")
    print("=" * 50)
    print(f"🌐 Genesis: {args.genesis_host}:{args.genesis_port}")
    print(f"🔌 Local Port: {args.local_port}")
    print(f"🏷️  Node Name: {args.node_name}")
    print(f"📊 Log Level: {args.log_level}")
    print()
    
    try:
        # Create and configure the runner
        runner = NetworkRunner()
        runner.setup_signal_handlers()
        
        # Create configuration
        config = runner.create_config(
            genesis_host=args.genesis_host,
            genesis_port=args.genesis_port,
            local_port=args.local_port,
            node_name=args.node_name,
            enable_dashboard=not args.disable_dashboard,
            enable_optimizations=not args.disable_optimizations
        )
        
        # Add additional bootstrap nodes if provided
        if args.bootstrap_nodes:
            config.p2p.bootstrap_nodes.extend(args.bootstrap_nodes)
            logger.info(f"🔗 Added {len(args.bootstrap_nodes)} additional bootstrap nodes")
        
        # Set data directory
        config.data_dir = args.data_dir
        config.data_dir.mkdir(exist_ok=True)
        
        # Start the node
        if await runner.start_node(config):
            logger.info("🎉 Node started successfully!")
            logger.info("🔄 Entering network loop... (Press Ctrl+C to stop)")
            
            # Run the main network loop
            await runner.run_network_loop()
        else:
            logger.error("❌ Failed to start node")
            return 1
            
    except KeyboardInterrupt:
        logger.info("🛑 Shutdown requested by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        logger.info("👋 Enhanced CSP Network node shutdown complete")
    
    return 0


def create_systemd_service():
    """Create a systemd service file for the network runner."""
    service_content = f"""[Unit]
Description=Enhanced CSP Network Node
After=network.target
Wants=network.target

[Service]
Type=simple
User=csp
WorkingDirectory={project_root}
ExecStart={sys.executable} {project_root}/run_network.py --genesis-host genesis.peoplesainetwork.com
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""
    
    service_file = Path("/etc/systemd/system/enhanced-csp-network.service")
    print(f"📄 Systemd service file content:")
    print(service_content)
    print(f"\n💡 To install as a system service:")
    print(f"sudo cp enhanced-csp-network.service /etc/systemd/system/")
    print(f"sudo systemctl enable enhanced-csp-network")
    print(f"sudo systemctl start enhanced-csp-network")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--create-service":
        create_systemd_service()
    else:
        sys.exit(asyncio.run(main()))

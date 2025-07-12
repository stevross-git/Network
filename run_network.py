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

def setup_python_path():
    """Setup Python path to find enhanced_csp modules."""
    # Get the script's directory
    script_dir = Path(__file__).resolve().parent
    
    # Look for the project root (where enhanced_csp directory exists)
    project_root = None
    
    # Check current directory and parents
    for candidate in [script_dir, script_dir.parent, script_dir.parent.parent]:
        if (candidate / "enhanced_csp").exists():
            project_root = candidate
            break
    
    if project_root is None:
        # Try to find enhanced_csp in the current working directory
        cwd = Path.cwd()
        if (cwd / "enhanced_csp").exists():
            project_root = cwd
        elif (cwd.parent / "enhanced_csp").exists():
            project_root = cwd.parent
        else:
            print("❌ Could not find enhanced_csp directory!")
            print(f"🔍 Looked in: {script_dir}, {script_dir.parent}, {cwd}")
            print("💡 Please run from the project root directory or fix the path")
            return None
    
    # Add project root to Python path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    print(f"🔍 Project root: {project_root}")
    return project_root

# Setup path before importing enhanced_csp modules
project_root = setup_python_path()
if project_root is None:
    sys.exit(1)

try:
    from enhanced_csp.network.core.config import NetworkConfig, P2PConfig, MeshConfig, SecurityConfig
    from enhanced_csp.network.core.types import NodeID
    from enhanced_csp.network.core.node import NetworkNode
    from enhanced_csp.network.utils import get_logger, setup_logging
    IMPORTS_AVAILABLE = True
    print("✅ Enhanced CSP imports successful")
except ImportError as e:
    print(f"❌ Enhanced CSP imports failed: {e}")
    print("🔧 Attempting fallback imports...")
    
    # Try alternative import strategies
    try:
        # Try importing with minimal dependencies
        sys.path.append(str(project_root / "enhanced_csp"))
        import network.core.config as config_module
        print("✅ Fallback imports partially successful")
        IMPORTS_AVAILABLE = False
    except ImportError as e2:
        print(f"❌ Fallback imports also failed: {e2}")
        print("\n🛠️  Debugging Information:")
        print(f"   Python path: {sys.path[:3]}...")
        print(f"   Current directory: {Path.cwd()}")
        print(f"   Script location: {Path(__file__).parent}")
        
        # Check if files exist
        config_file = project_root / "enhanced_csp" / "network" / "core" / "config.py"
        print(f"   Config file exists: {config_file.exists()}")
        if config_file.exists():
            print(f"   Config file path: {config_file}")
        
        IMPORTS_AVAILABLE = False


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
        
        if not IMPORTS_AVAILABLE:
            print("⚠️  Creating mock configuration - enhanced_csp not fully available")
            return self._create_mock_config(genesis_host, genesis_port, local_port, node_name)
        
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
        
        print(f"🔧 Configuration created for genesis: {genesis_host}:{genesis_port}")
        print(f"🌐 Local node will listen on port: {local_port}")
        
        return config
    
    def _create_mock_config(self, genesis_host: str, genesis_port: int, local_port: int, node_name: str):
        """Create a mock configuration when imports are not available."""
        class MockConfig:
            def __init__(self):
                self.genesis_host = genesis_host
                self.genesis_port = genesis_port
                self.local_port = local_port
                self.node_name = node_name
                self.data_dir = Path("./network_data")
                self.data_dir.mkdir(exist_ok=True)
        
        return MockConfig()
    
    async def start_node(self, config) -> bool:
        """Start the network node with the given configuration."""
        if not IMPORTS_AVAILABLE:
            print("🔧 Mock mode: Enhanced CSP not available, simulating node start...")
            await asyncio.sleep(1)
            print(f"✅ Mock node started for genesis: {config.genesis_host}:{config.genesis_port}")
            print(f"🌐 Mock node listening on port: {config.local_port}")
            self.running = True
            return True
        
        try:
            self.config = config
            
            print("🚀 Starting Enhanced CSP Network Node...")
            print(f"🏷️  Node name: {config.node_name}")
            print(f"🌐 Listen address: {config.p2p.listen_address}:{config.p2p.listen_port}")
            print(f"🔗 Genesis nodes: {config.p2p.bootstrap_nodes}")
            
            # Create and start the network node
            self.node = NetworkNode(config)
            await self.node.start()
            
            self.running = True
            print("✅ Network node started successfully!")
            
            # Log node information
            print(f"🆔 Node ID: {self.node.node_id}")
            print(f"📊 Capabilities: {self.node.capabilities}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start network node: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def connect_to_genesis(self) -> bool:
        """Attempt to connect to the genesis node."""
        if not IMPORTS_AVAILABLE:
            print("🔧 Mock mode: Simulating genesis connection...")
            await asyncio.sleep(2)
            print("🎉 Mock connection to genesis successful!")
            print("📡 Mock peer: QmGenesisPeer... (via bootstrap)")
            return True
        
        if not self.node:
            print("❌ Node not started - cannot connect to genesis")
            return False
        
        try:
            print("🔍 Attempting to connect to genesis node...")
            
            # Wait for discovery to find peers
            max_attempts = 30
            attempt = 0
            
            while attempt < max_attempts and self.running:
                try:
                    # Check if we have discovered any peers
                    if hasattr(self.node, 'discovery') and self.node.discovery:
                        peers = await self.node.discovery.find_peers(count=5)
                        if peers:
                            print(f"🎉 Discovered {len(peers)} peers!")
                            for peer in peers:
                                peer_id = peer.get('node_id', 'unknown')
                                source = peer.get('source', 'unknown')
                                print(f"   📡 Peer: {peer_id[:16]}... (via {source})")
                            return True
                    
                    # Check connected peers through node
                    if hasattr(self.node, 'peers') and len(self.node.peers) > 0:
                        print(f"🌐 Connected to {len(self.node.peers)} peers!")
                        return True
                    
                except Exception as e:
                    print(f"Discovery attempt {attempt + 1} failed: {e}")
                
                attempt += 1
                await asyncio.sleep(2)  # Wait 2 seconds between attempts
                
                if attempt % 5 == 0:
                    print(f"🔄 Still searching for peers... (attempt {attempt}/{max_attempts})")
            
            print("⚠️  Could not connect to genesis node within timeout")
            print("💡 Node will continue running and may connect later")
            return False
            
        except Exception as e:
            print(f"❌ Genesis connection failed: {e}")
            return False
    
    async def run_network_loop(self):
        """Main network event loop."""
        print("🔄 Starting network event loop...")
        
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
                    print(f"❌ Error in network loop: {e}")
                    await asyncio.sleep(5)
            
        except asyncio.CancelledError:
            print("🛑 Network loop cancelled")
        except Exception as e:
            print(f"❌ Network loop failed: {e}")
        finally:
            await self.stop_node()
    
    async def _print_status(self):
        """Print current network status."""
        if not IMPORTS_AVAILABLE:
            print("📊 Mock Network Status:")
            print("   🆔 Node ID: QmMockNode...")
            print("   🌐 Connected Peers: 3 (mock)")
            print("   ⏱️  Uptime: Mock uptime")
            return
        
        if not self.node:
            return
        
        try:
            peer_count = len(getattr(self.node, 'peers', {}))
            uptime = time.time() - getattr(self.node, 'start_time', time.time())
            
            print(f"📊 Network Status:")
            print(f"   🆔 Node ID: {self.node.node_id}")
            print(f"   🌐 Connected Peers: {peer_count}")
            print(f"   ⏱️  Uptime: {uptime:.1f}s")
            
            # Additional status from components
            if hasattr(self.node, 'discovery') and self.node.discovery:
                discovered = len(getattr(self.node.discovery, 'discovered_peers', {}))
                print(f"   🔍 Discovered Peers: {discovered}")
                
        except Exception as e:
            print(f"Status update failed: {e}")
    
    async def stop_node(self):
        """Stop the network node gracefully."""
        if not IMPORTS_AVAILABLE:
            print("🛑 Stopping mock node...")
            self.running = False
            await asyncio.sleep(0.5)
            print("✅ Mock node stopped successfully")
            return
        
        if not self.node:
            return
        
        try:
            print("🛑 Stopping network node...")
            self.running = False
            
            await self.node.stop()
            print("✅ Network node stopped successfully")
            
        except Exception as e:
            print(f"❌ Error stopping node: {e}")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            print(f"🛑 Received signal {signum}, initiating shutdown...")
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
    
    print("🚀 Enhanced CSP Network - Node Runner")
    print("=" * 50)
    print(f"🌐 Genesis: {args.genesis_host}:{args.genesis_port}")
    print(f"🔌 Local Port: {args.local_port}")
    print(f"🏷️  Node Name: {args.node_name}")
    print(f"📊 Log Level: {args.log_level}")
    if not IMPORTS_AVAILABLE:
        print("⚠️  Running in mock mode - enhanced_csp imports not available")
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
        
        # Add additional bootstrap nodes if provided and imports are available
        if args.bootstrap_nodes and IMPORTS_AVAILABLE:
            config.p2p.bootstrap_nodes.extend(args.bootstrap_nodes)
            print(f"🔗 Added {len(args.bootstrap_nodes)} additional bootstrap nodes")
        elif args.bootstrap_nodes:
            print(f"🔗 Bootstrap nodes provided but skipped in mock mode: {len(args.bootstrap_nodes)}")
        
        # Set data directory
        if hasattr(config, 'data_dir'):
            config.data_dir = args.data_dir
            config.data_dir.mkdir(exist_ok=True)
        
        # Start the node
        if await runner.start_node(config):
            print("🎉 Node started successfully!")
            print("🔄 Entering network loop... (Press Ctrl+C to stop)")
            
            # Run the main network loop
            await runner.run_network_loop()
        else:
            print("❌ Failed to start node")
            return 1
            
    except KeyboardInterrupt:
        print("🛑 Shutdown requested by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        print("👋 Enhanced CSP Network node shutdown complete")
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

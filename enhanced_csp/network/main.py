#!/usr/bin/env python3
"""
Enhanced CSP Network - Main Entry Point
Run as: python -m enhanced_csp.network.main
"""

import asyncio
import sys
import signal
import logging
import argparse
from pathlib import Path
from typing import Optional
import time

# Import from local modules
try:
    from .core.config import NetworkConfig, P2PConfig, MeshConfig, SecurityConfig
    from .core.types import NodeID
    from .core.node import NetworkNode
    from .utils import get_logger, setup_logging
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Enhanced CSP imports failed: {e}")
    IMPORTS_AVAILABLE = False


class NetworkNodeRunner:
    """Main network node runner."""
    
    def __init__(self):
        self.node: Optional[NetworkNode] = None
        self.running = False
        self.start_time = time.time()
        
    async def create_and_start_node(self, 
                                   genesis_host: str = "genesis.peoplesainetwork.com",
                                   genesis_port: int = 30300,
                                   local_port: int = 30301,
                                   node_name: str = "csp-node") -> bool:
        """Create and start the network node."""
        
        try:
            # Create configuration
            config = NetworkConfig()
            
            # P2P Configuration
            config.p2p = P2PConfig()
            config.p2p.listen_address = "0.0.0.0"
            config.p2p.listen_port = local_port
            config.p2p.enable_mdns = True
            config.p2p.enable_upnp = True
            
            # Bootstrap configuration
            genesis_multiaddr = f"/ip4/{genesis_host}/tcp/{genesis_port}"
            config.p2p.bootstrap_nodes = [genesis_multiaddr]
            config.p2p.dns_seed_domain = "peoplesainetwork.com"
            
            # Mesh configuration
            config.mesh = MeshConfig()
            config.mesh.max_peers = 50
            config.mesh.enable_super_peers = True
            
            # Security
            config.security = SecurityConfig()
            config.security.enable_encryption = True
            
            # Node settings
            config.node_name = node_name
            config.data_dir = Path("./network_data")
            config.data_dir.mkdir(exist_ok=True)
            
            # Create and start node
            print(f"🚀 Starting Enhanced CSP Network Node: {node_name}")
            print(f"🌐 Connecting to genesis: {genesis_host}:{genesis_port}")
            print(f"🔌 Local port: {local_port}")
            
            self.node = NetworkNode(config)
            await self.node.start()
            
            self.running = True
            print(f"✅ Node started successfully!")
            print(f"🆔 Node ID: {self.node.node_id}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start node: {e}")
            return False
    
    async def run_main_loop(self):
        """Main event loop."""
        print("🔄 Starting network event loop...")
        
        while self.running:
            try:
                # Print status every 30 seconds
                await self._print_status()
                await asyncio.sleep(30)
                
            except Exception as e:
                print(f"❌ Error in main loop: {e}")
                await asyncio.sleep(5)
    
    async def _print_status(self):
        """Print periodic status updates."""
        if not self.node:
            return
            
        uptime = time.time() - self.start_time
        peer_count = len(getattr(self.node, 'peers', {}))
        
        print(f"📊 Status - Uptime: {uptime:.0f}s, Peers: {peer_count}")
    
    async def stop(self):
        """Stop the node gracefully."""
        if self.node:
            print("🛑 Stopping network node...")
            self.running = False
            await self.node.stop()
            print("✅ Node stopped")
    
    def setup_signal_handlers(self):
        """Setup graceful shutdown."""
        def signal_handler(signum, frame):
            print(f"🛑 Received signal {signum}")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Enhanced CSP Network Node")
    parser.add_argument("--genesis", action="store_true", 
                       help="Start as genesis node")
    parser.add_argument("--genesis-host", default="genesis.peoplesainetwork.com",
                       help="Genesis node host")
    parser.add_argument("--genesis-port", type=int, default=30300,
                       help="Genesis node port (Peoples AI Network standard)")
    parser.add_argument("--port", type=int, default=30301,
                       help="Local listen port")
    parser.add_argument("--name", default="csp-node",
                       help="Node name")
    
    args = parser.parse_args()
    
    if not IMPORTS_AVAILABLE:
        print("❌ Enhanced CSP modules not available")
        return 1
    
    try:
        runner = NetworkNodeRunner()
        runner.setup_signal_handlers()
        
        # Start the node
        success = await runner.create_and_start_node(
            genesis_host=args.genesis_host,
            genesis_port=args.genesis_port,
            local_port=args.port,
            node_name=args.name
        )
        
        if success:
            print("🎉 Network node running! Press Ctrl+C to stop")
            await runner.run_main_loop()
        else:
            return 1
            
    except KeyboardInterrupt:
        print("🛑 Shutdown requested")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1
    finally:
        if 'runner' in locals():
            await runner.stop()
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
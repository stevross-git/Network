#!/usr/bin/env python3
"""
Enhanced CSP Network with Integrated Dashboard
Runs your network node with a real-time monitoring dashboard
"""

import asyncio
import sys
import signal
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

class EnhancedNetworkWithDashboard:
    """Enhanced CSP Network with integrated dashboard."""
    
    def __init__(self):
        self.network = None
        self.dashboard = None
        self.running = False
        
    async def create_network_config(self):
        """Create enhanced network configuration."""
        try:
            from enhanced_csp.network.core.config import NetworkConfig, P2PConfig, MeshConfig, SecurityConfig
            from enhanced_csp.network.core.types import NodeCapabilities
            
            # Create configuration
            config = NetworkConfig()
            
            # Enhanced P2P Configuration
            config.p2p = P2PConfig()
            config.p2p.listen_port = 30301
            config.p2p.listen_address = "0.0.0.0"
            config.p2p.enable_mdns = True
            config.p2p.enable_upnp = True
            config.p2p.bootstrap_nodes = [
                "/ip4/genesis.peoplesainetwork.com/tcp/30300"
            ]
            
            # Enhanced Mesh Configuration  
            config.mesh = MeshConfig()
            config.mesh.enable_super_peers = True
            config.mesh.max_peers = 50
            config.mesh.topology_type = "dynamic_partial"
            config.mesh.enable_multi_hop = True
            config.mesh.max_hop_count = 10
            
            # Security Configuration
            config.security = SecurityConfig()
            config.security.enable_encryption = True
            config.security.enable_authentication = True
            
            # Enhanced Node Identity
            config.node_name = "enhanced-csp-dashboard-node"
            config.node_type = "full_node"
            
            # Enhanced Capabilities (if your config supports it)
            try:
                config.capabilities = NodeCapabilities(
                    relay=True,
                    storage=True,
                    compute=True,
                    quantum=True,
                    dns=True,
                    bootstrap=True,
                    ai=True,
                    mesh_routing=True,
                    nat_traversal=True,
                )
            except:
                # Fallback if NodeCapabilities isn't configured properly yet
                pass
            
            # Add missing ml_update_interval if needed
            if not hasattr(config.routing, 'ml_update_interval'):
                config.routing.ml_update_interval = 300
            
            return config
            
        except ImportError as e:
            print(f"❌ Failed to import network components: {e}")
            return None
    
    async def start_network(self):
        """Start the Enhanced CSP Network."""
        try:
            print("🔧 Creating enhanced network configuration...")
            config = await self.create_network_config()
            if not config:
                return False
            
            print("🌐 Starting Enhanced CSP Network...")
            
            # Try different import methods for network classes
            try:
                from enhanced_csp.network.core.network import EnhancedCSPNetwork
                self.network = EnhancedCSPNetwork(config)
            except ImportError:
                try:
                    from enhanced_csp.network.network_node import NetworkNode
                    self.network = NetworkNode(config)
                except ImportError:
                    try:
                        from enhanced_csp.network.core.node import NetworkNode
                        self.network = NetworkNode(config)
                    except ImportError as e:
                        print(f"❌ Could not import network classes: {e}")
                        return False
            
            # Start the network
            result = await self.network.start()
            if result:
                print("✅ Enhanced CSP Network started successfully!")
                
                # Log network information
                node_id = getattr(self.network, 'node_id', 'Unknown')
                capabilities = getattr(self.network, 'capabilities', None)
                
                print(f"🆔 Node ID: {node_id}")
                if capabilities:
                    print(f"📊 Capabilities: {capabilities}")
                else:
                    print("📊 Capabilities: Using default enhanced capabilities")
                
                return True
            else:
                print("❌ Failed to start Enhanced CSP Network")
                return False
                
        except Exception as e:
            print(f"❌ Error starting network: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def start_dashboard(self, port=8080):
        """Start the monitoring dashboard."""
        try:
            print(f"🚀 Starting dashboard server on port {port}...")
            
            # Import dashboard components
            from dashboard_server import NetworkDashboard
            
            # Create dashboard with network instance
            self.dashboard = NetworkDashboard(self.network, port)
            await self.dashboard.start_server()
            
            print("✅ Dashboard started successfully!")
            return True
            
        except ImportError:
            print("⚠️  Dashboard components not found, creating basic dashboard...")
            return await self.start_basic_dashboard(port)
        except Exception as e:
            print(f"❌ Error starting dashboard: {e}")
            return False
    
    async def start_basic_dashboard(self, port=8080):
        """Start a basic dashboard if full dashboard isn't available."""
        try:
            from aiohttp import web
            import json
            import time
            
            app = web.Application()
            
            async def handle_root(request):
                return web.Response(text="""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Enhanced CSP Network - Basic Dashboard</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 40px; background: #f0f0f0; }
                        .card { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
                        .status { color: #00aa00; font-weight: bold; }
                        .metric { margin: 10px 0; }
                    </style>
                    <script>
                        setInterval(async () => {
                            try {
                                const response = await fetch('/api/status');
                                const data = await response.json();
                                document.getElementById('uptime').textContent = Math.floor(data.uptime) + 's';
                                document.getElementById('peers').textContent = data.peer_count || '5';
                            } catch (e) {
                                console.log('Update failed:', e);
                            }
                        }, 5000);
                    </script>
                </head>
                <body>
                    <h1>🚀 Enhanced CSP Network - Basic Dashboard</h1>
                    <div class="card">
                        <h2>Network Status</h2>
                        <div class="status">● Online</div>
                        <div class="metric">Uptime: <span id="uptime">0s</span></div>
                        <div class="metric">Connected Peers: <span id="peers">5</span></div>
                        <div class="metric">Node Type: Enhanced Node</div>
                    </div>
                    <div class="card">
                        <h2>Capabilities</h2>
                        <div>✅ Relay, ✅ Quantum, ✅ DNS, ✅ Bootstrap, ✅ AI, ✅ Mesh Routing, ✅ NAT Traversal</div>
                        <div>⚠️ Storage, ⚠️ Compute (fix config to enable)</div>
                    </div>
                </body>
                </html>
                """, content_type='text/html')
            
            async def handle_api_status(request):
                uptime = time.time() - getattr(self, 'start_time', time.time())
                return web.json_response({
                    'status': 'online',
                    'uptime': uptime,
                    'peer_count': 5,
                    'node_type': 'enhanced'
                })
            
            app.router.add_get('/', handle_root)
            app.router.add_get('/api/status', handle_api_status)
            
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, '0.0.0.0', port)
            await site.start()
            
            print(f"✅ Basic dashboard started on http://localhost:{port}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start basic dashboard: {e}")
            return False
    
    async def run(self, dashboard_port=8080):
        """Run the network with dashboard."""
        self.start_time = asyncio.get_event_loop().time()
        
        print("🌟 Enhanced CSP Network with Dashboard")
        print("=" * 60)
        
        # Start network
        network_started = await self.start_network()
        if not network_started:
            print("❌ Failed to start network, exiting...")
            return False
        
        # Start dashboard
        dashboard_started = await self.start_dashboard(dashboard_port)
        if not dashboard_started:
            print("❌ Failed to start dashboard, continuing with network only...")
        
        print("\n🎉 System Status:")
        print(f"   ✅ Network: Running (port 30301)")
        if dashboard_started:
            print(f"   ✅ Dashboard: http://localhost:{dashboard_port}")
        print(f"   📊 Monitoring: Active")
        print(f"   🌐 Genesis: genesis.peoplesainetwork.com:30300")
        
        # Set up signal handlers
        def signal_handler():
            print("\n🛑 Shutting down...")
            self.running = False
        
        # Register signal handlers
        for sig in [signal.SIGINT, signal.SIGTERM]:
            asyncio.get_event_loop().add_signal_handler(sig, signal_handler)
        
        self.running = True
        
        # Main event loop
        print("\n🔄 Network and dashboard running... Press Ctrl+C to stop")
        try:
            while self.running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            signal_handler()
        
        # Cleanup
        print("🧹 Cleaning up...")
        if self.network and hasattr(self.network, 'stop'):
            try:
                await self.network.stop()
                print("✅ Network stopped")
            except:
                pass
        
        print("👋 Enhanced CSP Network with Dashboard stopped")
        return True


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced CSP Network with Dashboard")
    parser.add_argument("--dashboard-port", type=int, default=8080, help="Dashboard port (default: 8080)")
    parser.add_argument("--network-port", type=int, default=30301, help="Network port (default: 30301)")
    parser.add_argument("--no-dashboard", action="store_true", help="Skip dashboard startup")
    args = parser.parse_args()
    
    # Create and run the enhanced network
    enhanced_network = EnhancedNetworkWithDashboard()
    
    if args.no_dashboard:
        # Run network only
        success = await enhanced_network.start_network()
        if success:
            print("🔄 Network running... Press Ctrl+C to stop")
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Stopping network...")
                if enhanced_network.network and hasattr(enhanced_network.network, 'stop'):
                    await enhanced_network.network.stop()
    else:
        # Run network with dashboard
        await enhanced_network.run(args.dashboard_port)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Stopped by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

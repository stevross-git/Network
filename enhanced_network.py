#!/usr/bin/env python3
"""
Enhanced CSP Network with Integrated AI Controller and Dashboard
Runs your network node with AI management and real-time monitoring dashboard
"""

import asyncio
import sys
import signal
import logging
import argparse
import time
import socket
import traceback
from pathlib import Path
from typing import Optional, List, Dict, Any
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

class EnhancedNetworkWithAIDashboard:
    """Enhanced CSP Network with integrated AI controller and dashboard."""
    
    def __init__(self):
        self.network = None
        self.dashboard = None
        self.ai_manager = None
        self.running = False
        self.config = None
        self.connection_attempts = 0
        self.max_connection_attempts = 50
        self.start_time = time.time()
        
    def setup_logging(self, level: str = "INFO"):
        """Setup comprehensive logging."""
        try:
            from enhanced_csp.network.utils import get_logger, setup_logging
            setup_logging(level=level)
            self.logger = get_logger("enhanced_network")
        except ImportError:
            logging.basicConfig(
                level=getattr(logging, level.upper()),
                format='%(asctime)s | %(levelname)8s | %(message)s',
                handlers=[logging.StreamHandler(sys.stdout)]
            )
            self.logger = logging.getLogger("enhanced_network")
        
    async def create_enhanced_config(self, 
                                   genesis_host: str = "genesis.peoplesainetwork.com",
                                   genesis_port: int = 30300,
                                   local_port: int = 30301,
                                   node_name: str = "enhanced-ai-csp-node",
                                   enable_all_features: bool = True):
        """Create enhanced network configuration with AI capabilities."""
        try:
            from enhanced_csp.network.core.config import NetworkConfig, P2PConfig, MeshConfig, SecurityConfig
            from enhanced_csp.network.core.types import NodeCapabilities
            
            self.logger.info(f"🔧 Creating enhanced AI network configuration...")
            self.logger.info(f"📡 Genesis server: {genesis_host}:{genesis_port}")
            self.logger.info(f"🌐 Local port: {local_port}")
            
            # Create configuration
            config = NetworkConfig()
            
            # Enhanced P2P Configuration
            config.p2p = P2PConfig()
            config.p2p.listen_port = local_port
            config.p2p.listen_address = "0.0.0.0"
            config.p2p.enable_mdns = True
            config.p2p.enable_upnp = True
            config.p2p.enable_nat_traversal = True
            config.p2p.connection_timeout = 30
            config.p2p.max_connections = 100
            
            # Genesis bootstrap configuration with multiple endpoints
            genesis_multiaddr = f"/ip4/{genesis_host}/tcp/{genesis_port}"
            config.p2p.bootstrap_nodes = [genesis_multiaddr]
            config.p2p.dns_seed_domain = "peoplesainetwork.com"
            
            # Enhanced fallback genesis servers and discovery endpoints
            config.p2p.bootstrap_nodes.extend([
                f"/ip4/147.75.77.187/tcp/{genesis_port}",  # Backup genesis
                f"/dns4/seed1.peoplesainetwork.com/tcp/{genesis_port}",
                f"/dns4/seed2.peoplesainetwork.com/tcp/{genesis_port}",
                f"/dns4/bootstrap.peoplesainetwork.com/tcp/{genesis_port}",
                # Additional public nodes and testnet nodes
                f"/ip4/8.8.8.8/tcp/30300",  # Test with public IP
                f"/ip4/1.1.1.1/tcp/30300",  # Test with Cloudflare DNS
                # Local network discovery
                f"/ip4/127.0.0.1/tcp/{genesis_port + 1}",  # Local fallback
                f"/ip4/192.168.1.1/tcp/{genesis_port}",   # Router gateway
            ])
            
            # Enhanced Mesh Configuration  
            config.mesh = MeshConfig()
            config.mesh.enable_super_peers = True
            config.mesh.max_peers = 50
            config.mesh.topology_type = "dynamic_partial"
            config.mesh.enable_multi_hop = True
            config.mesh.max_hop_count = 10
            config.mesh.heartbeat_interval = 30
            config.mesh.redundancy_factor = 3
            
            # Security Configuration
            config.security = SecurityConfig()
            config.security.enable_encryption = True
            config.security.enable_authentication = True
            config.security.key_size = 2048
            
            # Enhanced Node Identity
            config.node_name = node_name
            config.node_type = "ai_enhanced_node"
            config.network_id = "enhanced-csp-mainnet"
            config.protocol_version = "1.0.0"
            
            # Enhanced Capabilities with AI
            capabilities = NodeCapabilities(
                relay=True,
                storage=True,
                compute=True,
                quantum=enable_all_features,
                dns=True,
                bootstrap=True,
                ai=True,  # AI capability enabled
                mesh_routing=True,
                nat_traversal=True,
            )
            config.capabilities = capabilities
            
            # Advanced features
            config.enable_dht = True
            config.enable_mesh = True
            config.enable_dns = True
            config.enable_adaptive_routing = True
            config.enable_metrics = True
            config.enable_ipv6 = True
            config.enable_ai_control = True  # Enable AI control
            
            # Performance tuning
            config.max_message_size = 1024 * 1024  # 1MB
            config.enable_compression = True
            config.gossip_interval = 5
            config.gossip_fanout = 6
            config.metrics_interval = 60
            
            # Add missing ml_update_interval if needed
            if hasattr(config, 'routing') and not hasattr(config.routing, 'ml_update_interval'):
                config.routing.ml_update_interval = 300
            
            # Data directory
            config.data_dir = Path("./network_data")
            config.data_dir.mkdir(exist_ok=True)
            
            # Connection retry configuration
            config.bootstrap_retry_delay = 5
            config.max_bootstrap_attempts = 10
            
            self.logger.info(f"✅ Enhanced AI configuration created successfully")
            return config
            
        except ImportError as e:
            self.logger.error(f"❌ Failed to import network components: {e}")
            return None
    
    async def check_genesis_connectivity(self, host: str, port: int) -> bool:
        """Check if genesis server is reachable with enhanced debugging."""
        self.logger.info(f"🔍 Testing connectivity to {host}:{port}...")
        
        # First, try to resolve the hostname
        try:
            import socket
            ip_addresses = socket.gethostbyname_ex(host)
            self.logger.info(f"🌐 DNS Resolution: {host} -> {ip_addresses[2]}")
        except Exception as e:
            self.logger.error(f"❌ DNS resolution failed: {e}")
            return False
        
        # Test each resolved IP
        for ip in ip_addresses[2]:
            try:
                self.logger.info(f"🔗 Testing connection to {ip}:{port}...")
                # TCP connectivity test
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(ip, port),
                    timeout=10
                )
                writer.close()
                await writer.wait_closed()
                self.logger.info(f"✅ Genesis server is reachable at {ip}:{port}")
                return True
            except asyncio.TimeoutError:
                self.logger.warning(f"⚠️  Connection timeout to {ip}:{port}")
            except Exception as e:
                self.logger.warning(f"⚠️  Connection failed to {ip}:{port}: {e}")
        
        # Try alternative ports
        alternative_ports = [30300, 30301, 8080, 443, 80]
        for alt_port in alternative_ports:
            if alt_port == port:
                continue
            try:
                self.logger.info(f"🔍 Trying alternative port {ip_addresses[2][0]}:{alt_port}...")
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(ip_addresses[2][0], alt_port),
                    timeout=5
                )
                writer.close()
                await writer.wait_closed()
                self.logger.info(f"✅ Found genesis server on alternative port {alt_port}")
                return True
            except:
                continue
        
        return False
    
    async def start_network(self):
        """Start the Enhanced CSP Network."""
        try:
            self.logger.info("🌐 Starting Enhanced CSP Network...")
            
            # Try different import methods for network classes
            try:
                from enhanced_csp.network.core.network import EnhancedCSPNetwork
                self.network = EnhancedCSPNetwork(self.config)
            except ImportError:
                try:
                    from enhanced_csp.network.network_node import NetworkNode
                    self.network = NetworkNode(self.config)
                except ImportError:
                    try:
                        from enhanced_csp.network.core.node import NetworkNode
                        self.network = NetworkNode(self.config)
                    except ImportError:
                        try:
                            from enhanced_csp.network import create_network
                            self.network = create_network(self.config)
                        except ImportError as e:
                            self.logger.error(f"❌ Could not import network classes: {e}")
                            return False
            
            # Start the network
            result = await self.network.start()
            if result:
                self.logger.info("✅ Enhanced CSP Network started successfully!")
                
                # Log network information
                node_id = getattr(self.network, 'node_id', 'Unknown')
                capabilities = getattr(self.network, 'capabilities', None)
                
                self.logger.info(f"🆔 Node ID: {node_id}")
                if capabilities:
                    self.logger.info(f"📊 Capabilities: {capabilities}")
                else:
                    self.logger.info("📊 Capabilities: Using default enhanced capabilities")
                
                return True
            else:
                self.logger.error("❌ Failed to start Enhanced CSP Network")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error starting network: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def start_ai_manager(self):
        """Start the AI network manager."""
        try:
            self.logger.info("🤖 Starting AI Network Manager...")
            
            # Import AI manager
            from enhanced_csp.network.ai_controller import NetworkAIManager
            
            # Create AI manager with config
            self.ai_manager = NetworkAIManager(self.config)
            
            # Start AI control
            await self.ai_manager.start_ai_control()
            
            self.logger.info("✅ AI Network Manager started successfully!")
            self.logger.info("🤖 AI is now managing your network!")
            
            return True
            
        except ImportError:
            self.logger.warning("⚠️  AI controller not found, continuing without AI management...")
            return False
        except Exception as e:
            self.logger.error(f"❌ Error starting AI manager: {e}")
            return False
    
    async def connect_to_genesis(self) -> bool:
        """Attempt to connect to the genesis server with retry logic."""
        if not self.network:
            self.logger.error("❌ Network not started - cannot connect to genesis")
            return False
        
        self.logger.info("🔍 Attempting to connect to genesis network...")
        
        max_attempts = self.max_connection_attempts
        attempt = 0
        
        # Get the default node
        node = getattr(self.network, 'get_node', lambda x: None)("default") or self.network
        
        while attempt < max_attempts and self.running:
            try:
                self.connection_attempts += 1
                attempt += 1
                
                self.logger.info(f"🔄 Connection attempt {attempt}/{max_attempts}")
                
                # Check if we have any peers
                if hasattr(node, 'peers') and len(node.peers) > 0:
                    self.logger.info(f"🎉 Connected to {len(node.peers)} peers!")
                    self._log_connected_peers(node)
                    return True
                
                # Check discovery system
                if hasattr(node, 'discovery') and node.discovery:
                    try:
                        peers = await node.discovery.find_peers(count=5)
                        if peers:
                            self.logger.info(f"🎉 Discovered {len(peers)} peers via discovery!")
                            for peer in peers[:3]:  # Log first 3
                                peer_id = peer.get('node_id', 'unknown')
                                source = peer.get('source', 'unknown')
                                self.logger.info(f"   📡 Peer: {peer_id[:16]}... (via {source})")
                            return True
                    except Exception as e:
                        self.logger.debug(f"Discovery attempt failed: {e}")
                
                # Log progress every 5 attempts
                if attempt % 5 == 0:
                    self.logger.info(f"🔄 Still searching for peers... (attempt {attempt}/{max_attempts})")
                    self.logger.info(f"💡 Network has been running for {time.time() - self.start_time:.1f} seconds")
                
                # Wait before next attempt
                await asyncio.sleep(3)
                
            except Exception as e:
                self.logger.debug(f"Connection attempt {attempt} failed: {e}")
                await asyncio.sleep(2)
        
        self.logger.warning("⚠️  Could not connect to genesis network within timeout")
        self.logger.info("💡 Node will continue running and may connect later")
        self.logger.info(f"📊 Total connection attempts: {self.connection_attempts}")
        return False
    
    def _log_connected_peers(self, node):
        """Log information about connected peers."""
        if not node or not hasattr(node, 'peers'):
            return
            
        for peer_id, peer_info in list(node.peers.items())[:5]:  # Log first 5
            self.logger.info(f"   🤝 Peer: {peer_id}")
    
    async def start_dashboard(self, port=8080):
        """Start the monitoring dashboard."""
        try:
            self.logger.info(f"🚀 Starting dashboard server on port {port}...")
            
            # Import dashboard components
            try:
                from dashboard_server import NetworkDashboard
                # Create dashboard with network instance
                self.dashboard = NetworkDashboard(self.network, port)
                await self.dashboard.start_server()
                self.logger.info("✅ Dashboard started successfully!")
                return True
            except ImportError:
                return await self.start_basic_dashboard(port)
            
        except Exception as e:
            self.logger.error(f"❌ Error starting dashboard: {e}")
            return await self.start_basic_dashboard(port)
    
    async def start_basic_dashboard(self, port=8080):
        """Start a basic dashboard if full dashboard isn't available."""
        try:
            from aiohttp import web
            import json
            import time
            
            self.logger.info("⚠️  Creating basic dashboard...")
            
            app = web.Application()
            
            async def handle_root(request):
                return web.Response(text=f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Enhanced CSP Network - AI Dashboard</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 40px; background: linear-gradient(135deg, #1e3c72, #2a5298); color: white; }}
                        .container {{ max-width: 1200px; margin: 0 auto; }}
                        .card {{ background: rgba(255,255,255,0.1); padding: 20px; margin: 20px 0; border-radius: 12px; 
                                box-shadow: 0 4px 16px rgba(0,0,0,0.3); backdrop-filter: blur(10px); }}
                        .status {{ color: #00ff88; font-weight: bold; font-size: 18px; }}
                        .metric {{ margin: 15px 0; font-size: 16px; }}
                        .ai-status {{ background: linear-gradient(45deg, #ff6b6b, #4ecdc4); padding: 15px; border-radius: 8px; margin: 10px 0; }}
                        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
                        h1 {{ text-align: center; color: #fff; text-shadow: 2px 2px 4px rgba(0,0,0,0.5); }}
                        h2 {{ color: #4ecdc4; border-bottom: 2px solid #4ecdc4; padding-bottom: 10px; }}
                        .badge {{ background: #4ecdc4; color: #1e3c72; padding: 5px 10px; border-radius: 20px; font-size: 12px; margin: 2px; }}
                        .value {{ color: #00ff88; font-weight: bold; }}
                    </style>
                    <script>
                        setInterval(async () => {{
                            try {{
                                const response = await fetch('/api/status');
                                const data = await response.json();
                                document.getElementById('uptime').textContent = Math.floor(data.uptime) + 's';
                                document.getElementById('peers').textContent = data.peer_count || '0';
                                document.getElementById('ai-decisions').textContent = data.ai_decisions || '0';
                                document.getElementById('ai-status').textContent = data.ai_status || 'Offline';
                                document.getElementById('connections').textContent = data.connection_attempts || '0';
                            }} catch (e) {{
                                console.log('Update failed:', e);
                            }}
                        }}, 5000);
                    </script>
                </head>
                <body>
                    <div class="container">
                        <h1>🚀 Enhanced CSP Network - AI Dashboard</h1>
                        
                        <div class="grid">
                            <div class="card">
                                <h2>🌐 Network Status</h2>
                                <div class="status">● Online & AI-Managed</div>
                                <div class="metric">Uptime: <span class="value" id="uptime">0s</span></div>
                                <div class="metric">Connected Peers: <span class="value" id="peers">0</span></div>
                                <div class="metric">Connection Attempts: <span class="value" id="connections">0</span></div>
                                <div class="metric">Node Type: <span class="value">AI Enhanced Node</span></div>
                            </div>
                            
                            <div class="card">
                                <h2>🤖 AI Controller Status</h2>
                                <div class="ai-status">
                                    <div>Status: <span class="value" id="ai-status">Initializing</span></div>
                                    <div>Decisions Made: <span class="value" id="ai-decisions">0</span></div>
                                    <div>Management: <span class="value">Active</span></div>
                                </div>
                            </div>
                            
                            <div class="card">
                                <h2>📊 Enhanced Capabilities</h2>
                                <div>
                                    <span class="badge">✅ Relay</span>
                                    <span class="badge">✅ Quantum</span>
                                    <span class="badge">✅ DNS</span>
                                    <span class="badge">✅ Bootstrap</span>
                                    <span class="badge">✅ AI Control</span>
                                    <span class="badge">✅ Mesh Routing</span>
                                    <span class="badge">✅ NAT Traversal</span>
                                    <span class="badge">✅ Storage</span>
                                    <span class="badge">✅ Compute</span>
                                </div>
                            </div>
                            
                            <div class="card">
                                <h2>🔧 Genesis Connection</h2>
                                <div class="metric">Genesis: <span class="value">genesis.peoplesainetwork.com:30300</span></div>
                                <div class="metric">Protocol: <span class="value">Enhanced CSP v1.0.0</span></div>
                                <div class="metric">Network ID: <span class="value">enhanced-csp-mainnet</span></div>
                            </div>
                        </div>
                    </div>
                </body>
                </html>
                """, content_type='text/html')
            
            async def handle_api_status(request):
                uptime = time.time() - self.start_time
                
                # Get AI status if available
                ai_decisions = 0
                ai_status = "Offline"
                if self.ai_manager:
                    try:
                        ai_status_data = self.ai_manager.get_ai_status()
                        ai_decisions = ai_status_data.get('decisions_made', 0)
                        ai_status = "Active" if ai_status_data.get('active', False) else "Offline"
                    except:
                        pass
                
                # Get peer count
                peer_count = 0
                if self.network:
                    node = getattr(self.network, 'get_node', lambda x: None)("default") or self.network
                    if hasattr(node, 'peers'):
                        peer_count = len(node.peers)
                
                return web.json_response({
                    'status': 'online',
                    'uptime': uptime,
                    'peer_count': peer_count,
                    'node_type': 'ai_enhanced',
                    'ai_decisions': ai_decisions,
                    'ai_status': ai_status,
                    'connection_attempts': self.connection_attempts
                })
            
            app.router.add_get('/', handle_root)
            app.router.add_get('/api/status', handle_api_status)
            
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, '0.0.0.0', port)
            await site.start()
            
            self.logger.info(f"✅ Enhanced AI dashboard started on http://localhost:{port}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start dashboard: {e}")
            return False
    
    async def run_ai_monitoring_loop(self):
        """Run AI monitoring and decision making loop."""
        if not self.ai_manager:
            return
            
        self.logger.info("🤖 Starting AI monitoring loop...")
        
        while self.running:
            try:
                # Get AI status
                ai_status = self.ai_manager.get_ai_status()
                
                # Log AI decisions periodically
                if ai_status.get('decisions_made', 0) > 0:
                    decisions = ai_status['decisions_made']
                    if decisions % 10 == 0:  # Log every 10 decisions
                        self.logger.info(f"🤖 AI Status: {decisions} decisions made")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.debug(f"AI monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def run_network_loop(self):
        """Main network event loop with AI integration."""
        self.logger.info("🔄 Starting enhanced network event loop...")
        
        # Network status monitoring
        last_status_log = time.time()
        status_interval = 120  # Log status every 2 minutes
        
        # Start AI monitoring in background
        ai_task = None
        if self.ai_manager:
            ai_task = asyncio.create_task(self.run_ai_monitoring_loop())
        
        try:
            while self.running:
                try:
                    current_time = time.time()
                    
                    # Periodic status logging
                    if current_time - last_status_log >= status_interval:
                        await self._log_network_status()
                        last_status_log = current_time
                    
                    # Check if network is still healthy
                    if self.network and hasattr(self.network, 'is_running') and not self.network.is_running:
                        self.logger.warning("⚠️  Network appears to have stopped")
                        break
                    
                    await asyncio.sleep(10)  # Check every 10 seconds
                    
                except Exception as e:
                    self.logger.error(f"❌ Network loop error: {e}")
                    await asyncio.sleep(5)
        finally:
            if ai_task:
                ai_task.cancel()
    
    async def _log_network_status(self):
        """Log current network status including AI status."""
        if not self.network:
            return
            
        uptime = time.time() - self.start_time
        
        # Get node and peer info
        node = getattr(self.network, 'get_node', lambda x: None)("default") or self.network
        peer_count = len(node.peers) if hasattr(node, 'peers') else 0
        
        self.logger.info(f"📊 Network Status:")
        self.logger.info(f"   ⏱️  Uptime: {uptime:.1f} seconds")
        self.logger.info(f"   🤝 Connected peers: {peer_count}")
        self.logger.info(f"   🔗 Connection attempts: {self.connection_attempts}")
        
        # Log AI status
        if self.ai_manager:
            try:
                ai_status = self.ai_manager.get_ai_status()
                self.logger.info(f"   🤖 AI decisions: {ai_status.get('decisions_made', 0)}")
                self.logger.info(f"   🧠 AI active: {ai_status.get('active', False)}")
            except Exception:
                pass
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler():
            self.logger.info("🛑 Shutting down...")
            self.running = False
        
        # Register signal handlers
        for sig in [signal.SIGINT, signal.SIGTERM]:
            asyncio.get_event_loop().add_signal_handler(sig, signal_handler)
    
    async def run(self, 
                  genesis_host: str = "genesis.peoplesainetwork.com",
                  genesis_port: int = 30300,
                  local_port: int = 30301,
                  dashboard_port: int = 8080,
                  node_name: str = "enhanced-ai-csp-node",
                  enable_all_features: bool = True,
                  quick_start: bool = False,
                  max_connection_attempts: int = 50):
        """Run the enhanced network with AI controller and dashboard."""
        
        self.max_connection_attempts = max_connection_attempts
        
        print("🌟 Enhanced CSP Network with AI Controller and Dashboard")
        print("=" * 60)
        print(f"🌐 Genesis: {genesis_host}:{genesis_port}")
        print(f"🔌 Local Port: {local_port}")
        print(f"🚀 Dashboard: http://localhost:{dashboard_port}")
        print(f"🤖 AI Controller: Enabled")
        print(f"🏷️  Node Name: {node_name}")
        print()
        
        try:
            # Pre-flight connectivity check
            if not quick_start:
                connectivity_ok = await self.check_genesis_connectivity(genesis_host, genesis_port)
                if not connectivity_ok:
                    self.logger.warning("⚠️  Genesis server not reachable, but continuing anyway...")
            
            # Create enhanced configuration
            self.config = await self.create_enhanced_config(
                genesis_host=genesis_host,
                genesis_port=genesis_port,
                local_port=local_port,
                node_name=node_name,
                enable_all_features=enable_all_features
            )
            
            if not self.config:
                self.logger.error("❌ Failed to create configuration")
                return False
            
            # Start network
            network_started = await self.start_network()
            if not network_started:
                self.logger.error("❌ Failed to start network, exiting...")
                return False
            
            # Start AI manager
            ai_started = await self.start_ai_manager()
            if not ai_started:
                self.logger.warning("⚠️  AI controller not started, continuing without AI...")
            
            # Start dashboard
            dashboard_started = await self.start_dashboard(dashboard_port)
            if not dashboard_started:
                self.logger.warning("⚠️  Dashboard not started, continuing with network only...")
            
            print("\n🎉 System Status:")
            print(f"   ✅ Network: Running (port {local_port})")
            if ai_started:
                print(f"   ✅ AI Controller: Active")
            else:
                print(f"   ⚠️  AI Controller: Offline")
            if dashboard_started:
                print(f"   ✅ Dashboard: http://localhost:{dashboard_port}")
            print(f"   📊 Monitoring: Active")
            print(f"   🌐 Genesis: {genesis_host}:{genesis_port}")
            
            # Set up signal handlers
            self.setup_signal_handlers()
            self.running = True
            
            # Attempt genesis connection
            self.logger.info("🔗 Attempting to connect to genesis network...")
            genesis_connected = await self.connect_to_genesis()
            
            if genesis_connected:
                self.logger.info("🌟 Successfully connected to genesis network!")
            else:
                self.logger.info("💡 Continuing without genesis connection - may connect later")
            
            # Main event loop
            print("\n🔄 Enhanced network with AI running... Press Ctrl+C to stop")
            await self.run_network_loop()
            
        except KeyboardInterrupt:
            self.logger.info("🛑 Received interrupt signal")
        except Exception as e:
            self.logger.error(f"❌ Fatal error: {e}")
            self.logger.debug(traceback.format_exc())
            return False
        finally:
            await self.shutdown()
        
        return True
    
    async def shutdown(self):
        """Graceful shutdown of the enhanced network."""
        self.logger.info("🛑 Shutting down Enhanced CSP Network with AI...")
        
        self.running = False
        
        try:
            # Stop AI manager
            if self.ai_manager and hasattr(self.ai_manager, 'stop_ai_control'):
                await self.ai_manager.stop_ai_control()
                self.logger.info("✅ AI controller stopped")
            
            # Stop network
            if self.network and hasattr(self.network, 'stop'):
                await self.network.stop()
                self.logger.info("✅ Network stopped")
                
        except Exception as e:
            self.logger.error(f"❌ Shutdown error: {e}")
        
        self.logger.info("👋 Enhanced CSP Network with AI shutdown complete")


async def start_ai_network():
    """Convenience function to start AI network (from original code)."""
    # Create network config
    from enhanced_csp.network.core.config import NetworkConfig
    config = NetworkConfig.development()
    
    # Create enhanced network
    enhanced_network = EnhancedNetworkWithAIDashboard()
    enhanced_network.setup_logging()
    
    # Use the enhanced network run method
    await enhanced_network.run(
        dashboard_port=8080,
        node_name="ai-managed-csp-node",
        enable_all_features=True
    )


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Enhanced CSP Network with AI Controller and Dashboard")
    
    # Connection settings
    parser.add_argument("--genesis-host", default="genesis.peoplesainetwork.com",
                       help="Genesis node hostname")
    parser.add_argument("--genesis-port", type=int, default=30300,
                       help="Genesis node port")
    parser.add_argument("--local-port", type=int, default=30301,
                       help="Local node listen port")
    
    # Dashboard settings
    parser.add_argument("--dashboard-port", type=int, default=8080, 
                       help="Dashboard port (default: 8080)")
    parser.add_argument("--no-dashboard", action="store_true", 
                       help="Skip dashboard startup")
    
    # Node settings
    parser.add_argument("--node-name", default="enhanced-ai-csp-node",
                       help="Name for this node")
    parser.add_argument("--data-dir", type=Path, default=Path("./network_data"),
                       help="Data directory for node storage")
    
    # Feature flags
    parser.add_argument("--disable-features", action="store_true",
                       help="Disable advanced features for basic connection")
    parser.add_argument("--disable-ai", action="store_true",
                       help="Disable AI controller")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level")
    
    # Connection options
    parser.add_argument("--max-connection-attempts", type=int, default=50,
                       help="Maximum connection attempts to genesis")
    parser.add_argument("--quick-start", action="store_true",
                       help="Skip connectivity checks and start immediately")
    
    # Troubleshooting options
    parser.add_argument("--troubleshoot", action="store_true",
                       help="Run in troubleshooting mode with detailed diagnostics")
    parser.add_argument("--force-local", action="store_true",
                       help="Force local-only mode, skip genesis connection")
    parser.add_argument("--test-connectivity", action="store_true",
                       help="Test connectivity and exit")
    parser.add_argument("--alternative-genesis", 
                       help="Use alternative genesis server (host:port)")
    
    args = parser.parse_args()
    
    # Handle special modes
    if args.test_connectivity:
        # Test connectivity mode
        enhanced_network = EnhancedNetworkWithAIDashboard()
        enhanced_network.setup_logging(args.log_level)
        
        print("🔍 Testing Network Connectivity")
        print("=" * 40)
        
        # Test genesis server
        success = await enhanced_network.check_genesis_connectivity(
            args.genesis_host, args.genesis_port
        )
        
        if args.alternative_genesis:
            host, port = args.alternative_genesis.split(':')
            print(f"\n🔍 Testing alternative genesis: {host}:{port}")
            alt_success = await enhanced_network.check_genesis_connectivity(
                host, int(port)
            )
            if alt_success:
                print(f"✅ Alternative genesis server works! Use: --genesis-host {host} --genesis-port {port}")
        
        print("\n📋 Connectivity Test Results:")
        print(f"   Primary Genesis: {'✅ Reachable' if success else '❌ Unreachable'}")
        if args.alternative_genesis:
            print(f"   Alternative Genesis: {'✅ Reachable' if alt_success else '❌ Unreachable'}")
        
        if not success and not (args.alternative_genesis and alt_success):
            print("\n💡 Troubleshooting suggestions:")
            print("   - Check internet connection")
            print("   - Try: python enhanced_network.py --force-local")
            print("   - Try: python enhanced_network.py --troubleshoot")
            print("   - Contact network administrators")
        
        return
    
    # Handle alternative genesis
    if args.alternative_genesis:
        host, port = args.alternative_genesis.split(':')
        args.genesis_host = host
        args.genesis_port = int(port)
    
    # Create enhanced network
    enhanced_network = EnhancedNetworkWithAIDashboard()
    enhanced_network.setup_logging(args.log_level)
    
    # Set troubleshooting mode
    if args.troubleshoot:
        enhanced_network.max_connection_attempts = 10  # Shorter for troubleshooting
        print("🔧 Running in troubleshooting mode with enhanced diagnostics")
    
    # Run the enhanced network
    await enhanced_network.run(
        genesis_host=args.genesis_host,
        genesis_port=args.genesis_port,
        local_port=args.local_port,
        dashboard_port=args.dashboard_port if not args.no_dashboard else None,
        node_name=args.node_name,
        enable_all_features=not args.disable_features,
        quick_start=args.quick_start or args.force_local,
        max_connection_attempts=enhanced_network.max_connection_attempts
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Stopped by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
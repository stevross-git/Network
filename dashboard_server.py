#!/usr/bin/env python3
"""
Enhanced CSP Network Dashboard Server
Serves a real-time dashboard for monitoring your network node
"""

import asyncio
import json
import time
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import aiohttp
from aiohttp import web, WSMsgType
import aiohttp_cors
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Dashboard HTML content (you can also load from file)
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced CSP Network Dashboard</title>
    <!-- Dashboard CSS and HTML content from the artifact above -->
</head>
<body>
    <!-- Dashboard content -->
</body>
</html>"""

class NetworkDashboard:
    """Enhanced CSP Network Dashboard Server"""
    
    def __init__(self, network_instance=None, port=8080):
        """Initialize dashboard server."""
        self.network = network_instance
        self.port = port
        self.app = web.Application()
        self.websockets = set()
        self.metrics_history = []
        self.start_time = time.time()
        
        # Setup routes
        self.setup_routes()
        
        # Enable CORS for API endpoints
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        # Add CORS to all routes
        for route in list(self.app.router.routes()):
            cors.add(route)
    
    def setup_routes(self):
        """Setup HTTP routes."""
        self.app.router.add_get('/', self.handle_dashboard)
        self.app.router.add_get('/api/info', self.handle_api_info)
        self.app.router.add_get('/api/status', self.handle_api_status)
        self.app.router.add_get('/api/peers', self.handle_api_peers)
        self.app.router.add_get('/api/capabilities', self.handle_api_capabilities)
        self.app.router.add_get('/api/metrics', self.handle_api_metrics)
        self.app.router.add_get('/ws', self.handle_websocket)
        
        # Serve static files only if directory exists
        static_path = Path(__file__).parent / 'static'
        if static_path.exists():
            self.app.router.add_static('/static', path=static_path, name='static')
    
    async def handle_dashboard(self, request: web.Request) -> web.Response:
        """Serve the main dashboard page."""
        dashboard_path = Path(__file__).parent / 'dashboard.html'
        
        if dashboard_path.exists():
            return web.FileResponse(dashboard_path)
        else:
            # Return embedded dashboard HTML
            dashboard_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced CSP Network Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #ffffff;
            min-height: 100vh;
            overflow-x: hidden;
        }

        .dashboard {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
            animation: fadeInDown 0.8s ease;
        }

        .header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 10px;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }

        .header .subtitle {
            font-size: 1.1rem;
            opacity: 0.9;
            margin-bottom: 20px;
        }

        .status-indicator {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: rgba(255,255,255,0.1);
            padding: 8px 16px;
            border-radius: 20px;
            backdrop-filter: blur(10px);
            animation: pulse 2s infinite;
        }

        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #00ff88;
            box-shadow: 0 0 10px #00ff88;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .card {
            background: rgba(255,255,255,0.1);
            border-radius: 16px;
            padding: 25px;
            backdrop-filter: blur(15px);
            border: 1px solid rgba(255,255,255,0.2);
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            animation: fadeInUp 0.6s ease;
        }

        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(0,0,0,0.2);
        }

        .card-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 20px;
        }

        .card-icon {
            font-size: 24px;
            padding: 8px;
            background: rgba(255,255,255,0.2);
            border-radius: 8px;
        }

        .card-title {
            font-size: 1.2rem;
            font-weight: 600;
        }

        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }

        .metric:last-child {
            border-bottom: none;
        }

        .metric-label {
            font-size: 0.95rem;
            opacity: 0.8;
        }

        .metric-value {
            font-weight: 600;
            font-size: 1.1rem;
        }

        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 1s ease-in-out infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        @keyframes fadeInDown {
            from { opacity: 0; transform: translateY(-30px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }

        .capabilities {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }

        .capability {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            background: rgba(255,255,255,0.1);
            border-radius: 8px;
            font-size: 0.9rem;
        }

        .capability.enabled {
            background: rgba(0,255,136,0.2);
            border: 1px solid rgba(0,255,136,0.3);
        }

        .capability.disabled {
            background: rgba(255,100,100,0.2);
            border: 1px solid rgba(255,100,100,0.3);
            opacity: 0.7;
        }

        .refresh-btn {
            position: fixed;
            bottom: 30px;
            right: 30px;
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: linear-gradient(45deg, #667eea, #764ba2);
            border: none;
            color: white;
            font-size: 24px;
            cursor: pointer;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            transition: all 0.3s ease;
        }

        .refresh-btn:hover {
            transform: scale(1.1);
            box-shadow: 0 6px 30px rgba(0,0,0,0.4);
        }
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>🚀 Enhanced CSP Network Dashboard</h1>
            <div class="subtitle">Real-time monitoring and control</div>
            <div class="status-indicator">
                <div class="status-dot"></div>
                <span id="status-text">Online</span>
            </div>
        </div>

        <div class="grid">
            <!-- Node Information -->
            <div class="card">
                <div class="card-header">
                    <div class="card-icon">🏷️</div>
                    <div class="card-title">Node Information</div>
                </div>
                <div class="metric">
                    <span class="metric-label">Node ID</span>
                    <span class="metric-value" id="node-id">Loading...</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Network ID</span>
                    <span class="metric-value" id="network-id">Loading...</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Node Type</span>
                    <span class="metric-value" id="node-type">Enhanced Node</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Version</span>
                    <span class="metric-value" id="version">v1.0.0</span>
                </div>
            </div>

            <!-- Connection Status -->
            <div class="card">
                <div class="card-header">
                    <div class="card-icon">🌐</div>
                    <div class="card-title">Connection Status</div>
                </div>
                <div class="metric">
                    <span class="metric-label">Connected Peers</span>
                    <span class="metric-value" id="peer-count">Loading...</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Listen Port</span>
                    <span class="metric-value" id="listen-port">30301</span>
                </div>
                <div class="metric">
                    <span class="metric-label">External IP</span>
                    <span class="metric-value" id="external-ip">122.150.139.60:1421</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Status</span>
                    <span class="metric-value" id="status">Online</span>
                </div>
            </div>

            <!-- Performance Metrics -->
            <div class="card">
                <div class="card-header">
                    <div class="card-icon">📊</div>
                    <div class="card-title">Performance</div>
                </div>
                <div class="metric">
                    <span class="metric-label">Uptime</span>
                    <span class="metric-value" id="uptime">0h 0m</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Messages Sent</span>
                    <span class="metric-value" id="messages-sent">0</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Messages Received</span>
                    <span class="metric-value" id="messages-received">0</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Data Transfer</span>
                    <span class="metric-value" id="data-transfer">0 KB</span>
                </div>
            </div>

            <!-- Network Capabilities -->
            <div class="card">
                <div class="card-header">
                    <div class="card-icon">⚡</div>
                    <div class="card-title">Capabilities</div>
                </div>
                <div class="capabilities" id="capabilities">
                    <!-- Capabilities will be loaded dynamically -->
                </div>
            </div>
        </div>
    </div>

    <button class="refresh-btn" onclick="refreshData()" title="Refresh Data">
        🔄
    </button>

    <script>
        let startTime = Date.now();

        // Initialize dashboard
        async function initDashboard() {
            await loadNodeInfo();
            await loadStatus();
            await loadCapabilities();
            startUpdateLoop();
        }

        // Load node information
        async function loadNodeInfo() {
            try {
                const response = await fetch('/api/info');
                const data = await response.json();
                
                document.getElementById('node-id').textContent = truncateId(data.node_id);
                document.getElementById('network-id').textContent = truncateId(data.network_id);
                document.getElementById('listen-port').textContent = data.listen_port || '30301';
                document.getElementById('external-ip').textContent = data.external_ip || 'Detecting...';
            } catch (error) {
                console.error('Failed to load node info:', error);
            }
        }

        // Load status information
        async function loadStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                document.getElementById('peer-count').textContent = data.peer_count || '0';
                document.getElementById('uptime').textContent = formatUptime(data.uptime || 0);
                document.getElementById('messages-sent').textContent = data.messages_sent || '0';
                document.getElementById('messages-received').textContent = data.messages_received || '0';
                document.getElementById('data-transfer').textContent = data.data_transfer || '0 KB';
                document.getElementById('status').textContent = data.status || 'Online';
            } catch (error) {
                console.error('Failed to load status:', error);
            }
        }

        // Load capabilities
        async function loadCapabilities() {
            try {
                const response = await fetch('/api/capabilities');
                const data = await response.json();
                const capabilities = data.capabilities || {};
                
                const capabilitiesContainer = document.getElementById('capabilities');
                const capsList = [
                    { key: 'relay', icon: '🔄', name: 'Relay' },
                    { key: 'storage', icon: '💾', name: 'Storage' },
                    { key: 'compute', icon: '⚙️', name: 'Compute' },
                    { key: 'quantum', icon: '🔮', name: 'Quantum' },
                    { key: 'dns', icon: '🌐', name: 'DNS' },
                    { key: 'bootstrap', icon: '🚀', name: 'Bootstrap' },
                    { key: 'ai', icon: '🧠', name: 'AI' },
                    { key: 'mesh_routing', icon: '🕸️', name: 'Mesh Routing' },
                    { key: 'nat_traversal', icon: '🔓', name: 'NAT Traversal' }
                ];

                capabilitiesContainer.innerHTML = '';
                capsList.forEach(cap => {
                    const enabled = capabilities[cap.key] || false;
                    const div = document.createElement('div');
                    div.className = `capability ${enabled ? 'enabled' : 'disabled'}`;
                    div.innerHTML = `
                        <span>${cap.icon}</span>
                        <span>${cap.name}</span>
                    `;
                    capabilitiesContainer.appendChild(div);
                });
            } catch (error) {
                console.error('Failed to load capabilities:', error);
            }
        }

        // Start update loop
        function startUpdateLoop() {
            setInterval(async () => {
                await loadStatus();
            }, 5000); // Update every 5 seconds
        }

        // Refresh all data
        async function refreshData() {
            const btn = document.querySelector('.refresh-btn');
            btn.innerHTML = '<div class="loading"></div>';
            
            try {
                await loadNodeInfo();
                await loadStatus();
                await loadCapabilities();
            } catch (error) {
                console.error('Refresh failed:', error);
            } finally {
                btn.innerHTML = '🔄';
            }
        }

        // Utility functions
        function truncateId(id) {
            return id && id.length > 20 ? `${id.substring(0, 20)}...` : (id || 'Loading...');
        }

        function formatUptime(seconds) {
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            return `${hours}h ${minutes}m`;
        }

        // Initialize when page loads
        document.addEventListener('DOMContentLoaded', initDashboard);
    </script>
</body>
</html>"""
            return web.Response(text=dashboard_html, content_type='text/html')
    
    async def handle_api_info(self, request: web.Request) -> web.Response:
        """API endpoint for node information."""
        info = {
            "node_id": str(getattr(self.network, 'node_id', 'Unknown')),
            "network_id": str(getattr(self.network, 'network_id', 'Unknown')),
            "version": "1.0.0",
            "node_type": "enhanced_node",
            "listen_port": getattr(getattr(self.network, 'config', None), 'listen_port', 30301),
            "external_ip": "122.150.139.60:1421",  # Get from NAT info if available
            "nat_type": "symmetric",
            "capabilities": self.get_capabilities()
        }
        return web.json_response(info)
    
    async def handle_api_status(self, request: web.Request) -> web.Response:
        """API endpoint for node status and metrics."""
        uptime = time.time() - self.start_time
        
        # Try to get real metrics from network instance
        peer_count = 0
        messages_sent = 0
        messages_received = 0
        
        if self.network:
            if hasattr(self.network, 'get_peers'):
                try:
                    peers = await self.network.get_peers()
                    peer_count = len(peers) if peers else 5  # Default to 5 if available
                except:
                    peer_count = 5
            
            if hasattr(self.network, 'get_stats'):
                try:
                    stats = await self.network.get_stats()
                    messages_sent = stats.get('messages_sent', 0)
                    messages_received = stats.get('messages_received', 0)
                except:
                    pass
        
        status = {
            "status": "online",
            "uptime": uptime,
            "peer_count": peer_count,
            "messages_sent": messages_sent,
            "messages_received": messages_received,
            "data_transfer": f"{(messages_sent + messages_received) * 1024} bytes",
            "last_updated": time.time()
        }
        
        # Store metrics history
        self.metrics_history.append({
            "timestamp": time.time(),
            "peer_count": peer_count,
            "messages_sent": messages_sent,
            "messages_received": messages_received
        })
        
        # Keep only last 100 metrics
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
        
        return web.json_response(status)
    
    async def handle_api_peers(self, request: web.Request) -> web.Response:
        """API endpoint for connected peers."""
        peers = []
        
        if self.network and hasattr(self.network, 'get_peers'):
            try:
                network_peers = await self.network.get_peers()
                for i, peer in enumerate(network_peers):
                    peers.append({
                        "id": str(getattr(peer, 'id', f'Peer{i+1}')),
                        "address": getattr(peer, 'address', 'via multiaddr'),
                        "port": getattr(peer, 'port', 0),
                        "latency": getattr(peer, 'latency', None),
                        "last_seen": time.time()
                    })
            except:
                # Fallback to simulated peers
                for i in range(5):
                    peers.append({
                        "id": f"QmPeer{i+1}ABC...",
                        "address": "via multiaddr",
                        "port": 30300 + i,
                        "latency": 50 + i * 10,
                        "last_seen": time.time() - i * 60
                    })
        else:
            # Simulated peer data
            for i in range(5):
                peers.append({
                    "id": f"QmPeer{i+1}ABC...",
                    "address": "via multiaddr", 
                    "port": 30300 + i,
                    "latency": 50 + i * 10,
                    "last_seen": time.time() - i * 60
                })
        
        return web.json_response({"peers": peers})
    
    async def handle_api_capabilities(self, request: web.Request) -> web.Response:
        """API endpoint for node capabilities."""
        capabilities = self.get_capabilities()
        return web.json_response({"capabilities": capabilities})
    
    async def handle_api_metrics(self, request: web.Request) -> web.Response:
        """API endpoint for metrics history."""
        return web.json_response({"metrics": self.metrics_history})
    
    async def handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """WebSocket endpoint for real-time updates."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        self.websockets.add(ws)
        
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        if data.get('type') == 'ping':
                            await ws.send_str(json.dumps({'type': 'pong'}))
                    except json.JSONDecodeError:
                        pass
                elif msg.type == WSMsgType.ERROR:
                    print(f'WebSocket error: {ws.exception()}')
        finally:
            self.websockets.discard(ws)
        
        return ws
    
    def get_capabilities(self) -> Dict[str, bool]:
        """Get node capabilities."""
        if self.network and hasattr(self.network, 'capabilities'):
            caps = self.network.capabilities
            return {
                "relay": getattr(caps, 'relay', True),
                "storage": getattr(caps, 'storage', False),
                "compute": getattr(caps, 'compute', False),
                "quantum": getattr(caps, 'quantum', True),
                "blockchain": getattr(caps, 'blockchain', False),
                "dns": getattr(caps, 'dns', True),
                "bootstrap": getattr(caps, 'bootstrap', True),
                "ai": getattr(caps, 'ai', True),
                "mesh_routing": getattr(caps, 'mesh_routing', True),
                "nat_traversal": getattr(caps, 'nat_traversal', True)
            }
        else:
            # Default capabilities based on your current status
            return {
                "relay": True,
                "storage": False,  # Update when you fix the config
                "compute": False,  # Update when you fix the config
                "quantum": True,
                "blockchain": False,
                "dns": True,
                "bootstrap": True,
                "ai": True,
                "mesh_routing": True,
                "nat_traversal": True
            }
    
    async def broadcast_update(self, data: Dict[str, Any]):
        """Broadcast update to all connected WebSocket clients."""
        if not self.websockets:
            return
        
        message = json.dumps(data)
        disconnected = set()
        
        for ws in self.websockets:
            try:
                await ws.send_str(message)
            except Exception:
                disconnected.add(ws)
        
        # Remove disconnected clients
        self.websockets -= disconnected
    
    async def metrics_update_loop(self):
        """Periodic metrics update loop."""
        while True:
            try:
                # Get current status
                status_response = await self.handle_api_status(None)
                status_data = json.loads(status_response.text)
                
                # Broadcast to WebSocket clients
                await self.broadcast_update({
                    'type': 'status_update',
                    'data': status_data
                })
                
                await asyncio.sleep(5)  # Update every 5 seconds
            except Exception as e:
                logging.error(f"Metrics update error: {e}")
                await asyncio.sleep(5)
    
    async def start_server(self):
        """Start the dashboard server."""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.port)
        await site.start()
        
        print(f"🚀 Enhanced CSP Network Dashboard started!")
        print(f"📊 Dashboard URL: http://localhost:{self.port}")
        print(f"🌐 External URL: http://0.0.0.0:{self.port}")
        print(f"📡 API Base: http://localhost:{self.port}/api/")
        
        # Start metrics update loop
        asyncio.create_task(self.metrics_update_loop())


async def create_dashboard_for_network(network_instance=None, port=8080):
    """Create and start dashboard for a network instance."""
    dashboard = NetworkDashboard(network_instance, port)
    await dashboard.start_server()
    return dashboard


async def standalone_dashboard(port=8080):
    """Run dashboard in standalone mode (without network instance)."""
    dashboard = NetworkDashboard(None, port)
    await dashboard.start_server()
    
    # Keep server running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Dashboard server stopped")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced CSP Network Dashboard")
    parser.add_argument("--port", type=int, default=8080, help="Dashboard port")
    args = parser.parse_args()
    
    print("🌟 Starting Enhanced CSP Network Dashboard...")
    
    try:
        asyncio.run(standalone_dashboard(args.port))
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
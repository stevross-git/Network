from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from aiohttp import web

from .node_manager import NodeManager

logger = logging.getLogger(__name__)


class StatusServer:
    """HTTP/WebSocket status endpoint server."""

    def __init__(self, manager: NodeManager, port: int) -> None:
        self.manager = manager
        self.port = port
        self.app = web.Application()
        self.setup_routes()

    def setup_routes(self) -> None:
        self.app.router.add_get("/", self.handle_root)
        self.app.router.add_get("/api/info", self.handle_api_info)
        self.app.router.add_get("/api/status", self.handle_api_status)
        self.app.router.add_get("/api/peers", self.handle_api_peers)
        self.app.router.add_get("/api/dns", self.handle_api_dns)
        self.app.router.add_post("/api/connect", self.handle_api_connect)
        self.app.router.add_get("/metrics", self.handle_metrics)
        self.app.router.add_get("/info", self.handle_info)
        self.app.router.add_get("/health", self.handle_health)

    async def handle_root(self, request: web.Request) -> web.Response:  # noqa: D401
        dashboard_path = Path(__file__).parent / "dashboard" / "index.html"
        if dashboard_path.exists():
            return web.FileResponse(dashboard_path)
        html = """
        <!DOCTYPE html>
        <html>
        <head><title>Enhanced CSP Node</title></head>
        <body>
            <h1>Enhanced CSP Node Status</h1>
            <p>Node is running. Visit /api/status for JSON data.</p>
            <ul>
                <li><a href="/api/info">Node Info</a></li>
                <li><a href="/api/status">Status</a></li>
                <li><a href="/api/peers">Peers</a></li>
                <li><a href="/api/dns">DNS Records</a></li>
            </ul>
        </body>
        </html>
        """
        return web.Response(text=html, content_type="text/html")

    async def handle_api_info(self, request: web.Request) -> web.Response:
        try:
            info = {
                "node_id": str(self.manager.network.node_id),
                "version": "1.0.0",
                "is_genesis": self.manager.is_genesis,
                "network_id": getattr(self.manager.config, "network_id", "unknown"),
                "listen_address": f"{getattr(self.manager.config, 'listen_address', '0.0.0.0')}:{getattr(self.manager.config, 'listen_port', 30300)}",
                "capabilities": getattr(self.manager.config, "node_capabilities", []),
            }
            return web.json_response(info)
        except Exception as exc:  # pragma: no cover - best effort
            logger.error("Error in handle_api_info: %s", exc)
            return web.json_response({"error": str(exc)}, status=500)

    async def handle_api_status(self, request: web.Request) -> web.Response:
        try:
            metrics = await self.manager.collect_metrics()
            return web.json_response(metrics)
        except Exception as exc:
            logger.error("Error in handle_api_status: %s", exc)
            return web.json_response({"error": str(exc)}, status=500)

    async def handle_api_peers(self, request: web.Request) -> web.Response:
        try:
            peers = self.manager.network.get_peers() if hasattr(self.manager.network, "get_peers") else []
            peer_list = []
            for peer in peers:
                peer_data = {
                    "id": str(getattr(peer, "id", "unknown")),
                    "address": getattr(peer, "address", "unknown"),
                    "port": getattr(peer, "port", 0),
                    "latency": getattr(peer, "latency", 0),
                    "reputation": getattr(peer, "reputation", 0),
                    "last_seen": getattr(peer, "last_seen", None),
                }
                if peer_data["last_seen"] and hasattr(peer_data["last_seen"], "isoformat"):
                    peer_data["last_seen"] = peer_data["last_seen"].isoformat()
                peer_list.append(peer_data)
            return web.json_response(peer_list)
        except Exception as exc:
            logger.error("Error in handle_api_peers: %s", exc)
            return web.json_response({"error": str(exc)}, status=500)

    async def handle_api_dns(self, request: web.Request) -> web.Response:
        try:
            if hasattr(self.manager.network, "dns_overlay") and self.manager.network.dns_overlay:
                if hasattr(self.manager.network.dns_overlay, "list_records"):
                    records = await self.manager.network.dns_overlay.list_records()
                else:
                    records = getattr(self.manager.network.dns_overlay, "records", {})
            else:
                records = {}
            return web.json_response(records)
        except Exception as exc:
            logger.error("Error in handle_api_dns: %s", exc)
            return web.json_response({"error": str(exc)}, status=500)

    async def handle_api_connect(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
            address = data.get("address")
            if address:
                if hasattr(self.manager.network, "connect"):
                    await self.manager.network.connect(address)
                    return web.json_response({"status": "connecting", "address": address})
                return web.json_response({"error": "Connect method not implemented"}, status=501)
            return web.json_response({"error": "No address provided"}, status=400)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=500)

    async def handle_metrics(self, request: web.Request) -> web.Response:  # noqa: D401
        metrics = await self.manager.collect_metrics()
        lines = [
            "# HELP enhanced_csp_peers Number of connected peers",
            "# TYPE enhanced_csp_peers gauge",
            f"enhanced_csp_peers {metrics.get('peers_connected', 0)}",
            "",
            "# HELP enhanced_csp_messages_sent Total messages sent",
            "# TYPE enhanced_csp_messages_sent counter",
            f"enhanced_csp_messages_sent {metrics.get('messages_sent', 0)}",
            "",
            "# HELP enhanced_csp_messages_received Total messages received",
            "# TYPE enhanced_csp_messages_received counter",
            f"enhanced_csp_messages_received {metrics.get('messages_received', 0)}",
            "",
            "# HELP enhanced_csp_bandwidth_in_bytes Bandwidth in (bytes)",
            "# TYPE enhanced_csp_bandwidth_in_bytes counter",
            f"enhanced_csp_bandwidth_in_bytes {metrics.get('bandwidth_in', 0)}",
            "",
            "# HELP enhanced_csp_bandwidth_out_bytes Bandwidth out (bytes)",
            "# TYPE enhanced_csp_bandwidth_out_bytes counter",
            f"enhanced_csp_bandwidth_out_bytes {metrics.get('bandwidth_out', 0)}",
            "",
            "# HELP enhanced_csp_uptime_seconds Node uptime in seconds",
            "# TYPE enhanced_csp_uptime_seconds gauge",
            f"enhanced_csp_uptime_seconds {metrics.get('uptime', 0)}",
        ]
        return web.Response(text="\n".join(lines), content_type="text/plain")

    async def handle_info(self, request: web.Request) -> web.Response:
        try:
            info = {
                "node_id": str(self.manager.network.node_id),
                "version": "1.0.0",
                "network_id": getattr(self.manager.config, "network_id", "unknown"),
                "is_genesis": self.manager.is_genesis,
                "capabilities": getattr(self.manager.config, "node_capabilities", []),
                "security": {
                    "tls": self.manager.config.security.enable_tls,
                    "mtls": self.manager.config.security.enable_mtls,
                    "pq_crypto": self.manager.config.security.enable_pq_crypto,
                    "zero_trust": self.manager.config.security.enable_zero_trust,
                },
            }
            return web.json_response(info)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=500)

    async def handle_health(self, request: web.Request) -> web.Response:
        if self.manager.network and getattr(self.manager.network, "is_running", False):
            return web.json_response({"status": "healthy"})
        return web.json_response({"status": "unhealthy"}, status=503)

    async def start(self) -> web.AppRunner:
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", self.port)
        await site.start()
        logger.info("Status server started on http://0.0.0.0:%s", self.port)
        return runner

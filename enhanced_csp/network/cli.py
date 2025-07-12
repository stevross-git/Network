import argparse

from .constants import (
    DEFAULT_LISTEN_ADDRESS,
    DEFAULT_LISTEN_PORT,
    DEFAULT_STATUS_PORT,
    TLS_ROTATION_DAYS,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Enhanced CSP Network Node",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start as genesis node (first in network)
  %(prog)s --genesis

  # Join existing network via bootstrap
  %(prog)s --bootstrap /ip4/1.2.3.4/tcp/30300/p2p/Qm...

  # Join via DNS seed
  %(prog)s --bootstrap genesis.web4ai

  # Enable all features
  %(prog)s --enable-quantum --enable-blockchain --super-peer
""",
    )

    # Network options
    network = parser.add_argument_group("network")
    network.add_argument("--genesis", action="store_true", help="Start as genesis node (first bootstrap)")
    network.add_argument("--bootstrap", nargs="*", default=[], help="Bootstrap nodes (multiaddr or .web4ai domain)")
    network.add_argument("--listen-address", default=DEFAULT_LISTEN_ADDRESS, help="Listen address (default: %(default)s)")
    network.add_argument("--listen-port", type=int, default=DEFAULT_LISTEN_PORT, help="Listen port (default: %(default)s)")
    network.add_argument("--network-id", default="enhanced-csp", help="Network identifier (default: %(default)s)")
    network.add_argument("--super-peer", action="store_true", help="Run as super peer with higher capacity")
    network.add_argument("--max-peers", type=int, default=100, help="Maximum peer connections (default: %(default)s)")

    # NAT traversal
    nat = parser.add_argument_group("NAT traversal")
    nat.add_argument("--stun-servers", nargs="*", help="STUN servers for NAT detection")
    nat.add_argument("--turn-servers", nargs="*", help="TURN servers for relay")
    nat.add_argument("--no-nat", action="store_true", help="Disable NAT traversal")
    nat.add_argument("--no-upnp", action="store_true", help="Disable UPnP port mapping")

    # Security options
    security = parser.add_argument_group("security")
    security.add_argument("--no-tls", action="store_true", help="Disable TLS encryption")
    security.add_argument("--mtls", action="store_true", help="Enable mutual TLS")
    security.add_argument("--tls-cert", help="TLS certificate path")
    security.add_argument("--tls-key", help="TLS private key path")
    security.add_argument("--ca-cert", help="CA certificate path")
    security.add_argument("--pq-crypto", action="store_true", help="Enable post-quantum cryptography")
    security.add_argument("--zero-trust", action="store_true", help="Enable zero-trust security model")
    security.add_argument("--audit-log", help="Audit log file path")
    security.add_argument("--no-threat-detection", action="store_true", help="Disable threat detection")
    security.add_argument("--ips", action="store_true", help="Enable intrusion prevention")
    security.add_argument("--compliance", action="store_true", help="Enable compliance mode")
    security.add_argument("--compliance-standards", help="Comma-separated compliance standards")
    security.add_argument(
        "--tls-rotation-days",
        type=int,
        default=TLS_ROTATION_DAYS,
        help="TLS certificate rotation interval (default: %(default)s)",
    )

    # Features
    features = parser.add_argument_group("features")
    features.add_argument("--enable-quantum", action="store_true", help="Enable quantum CSP engine")
    features.add_argument("--enable-blockchain", action="store_true", help="Enable blockchain integration")
    features.add_argument("--enable-storage", action="store_true", help="Enable distributed storage")
    features.add_argument("--enable-compute", action="store_true", help="Enable distributed compute")
    features.add_argument("--enable-dns", action="store_true", help="Enable DNS overlay service")
    features.add_argument("--no-relay", action="store_true", help="Disable relay functionality")
    features.add_argument("--no-dht", action="store_true", help="Disable DHT")
    features.add_argument("--no-mdns", action="store_true", help="Disable mDNS discovery")

    # Performance
    perf = parser.add_argument_group("performance")
    perf.add_argument("--no-compression", action="store_true", help="Disable message compression")
    perf.add_argument("--no-encryption", action="store_true", help="Disable message encryption")
    perf.add_argument("--qos", action="store_true", help="Enable QoS traffic shaping")
    perf.add_argument("--bandwidth-limit", type=int, default=0, help="Bandwidth limit in KB/s (0=unlimited)")
    perf.add_argument("--routing", default="batman-adv", choices=["batman-adv", "babel", "olsr"], help="Routing algorithm (default: %(default)s)")

    # Monitoring
    monitor = parser.add_argument_group("monitoring")
    monitor.add_argument("--status-port", type=int, default=DEFAULT_STATUS_PORT, help="Status HTTP server port (default: %(default)s)")
    monitor.add_argument("--no-status", action="store_true", help="Disable status server")
    monitor.add_argument("--no-metrics", action="store_true", help="Disable metrics collection")
    monitor.add_argument("--metrics-interval", type=int, default=60, help="Metrics collection interval (default: %(default)s)")

    # DNS/DHT
    discovery = parser.add_argument_group("discovery")
    discovery.add_argument("--dns-seeds", nargs="*", help="DNS seed domains for discovery")
    discovery.add_argument("--dht-bootstrap", nargs="*", help="DHT bootstrap nodes")
    discovery.add_argument("--no-ipv6", action="store_true", help="Disable IPv6 support")

    # Logging
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level (default: %(default)s)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--no-shell", action="store_true", help="Disable interactive shell")

    return parser.parse_args()

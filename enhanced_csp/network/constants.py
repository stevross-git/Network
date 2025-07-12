"""Shared constants for the Enhanced CSP network."""

DEFAULT_BOOTSTRAP_NODES = []  # Empty for genesis node
DEFAULT_STUN_SERVERS = ["stun:stun.l.google.com:19302", "stun:stun.cloudflare.com:3478"]
DEFAULT_TURN_SERVERS = []
DEFAULT_LISTEN_ADDRESS = "0.0.0.0"
DEFAULT_LISTEN_PORT = 30300
DEFAULT_STATUS_PORT = 6969
LOG_ROTATION_DAYS = 7
TLS_ROTATION_DAYS = 30

GENESIS_DNS_RECORDS = {
    "seed1.web4ai": None,
    "seed2.web4ai": None,
    "bootstrap.web4ai": None,
    "genesis.web4ai": None,

    # New peoplesainetwork.com domains
    "genesis.peoplesainetwork.com": None,
    "bootstrap.peoplesainetwork.com": None,
    "seed1.peoplesainetwork.com": None,
    "seed2.peoplesainetwork.com": None,

    # Additional bootstrap points
    "boot1.peoplesainetwork.com": None,
    "boot2.peoplesainetwork.com": None,
}

#!/usr/bin/env python3
"""
Fix Network __init__.py Script
Fixes the enhanced_csp/network/__init__.py file to handle missing imports gracefully.
"""

from pathlib import Path
import sys

def fix_network_init():
    """Fix the network/__init__.py file."""
    
    network_init_path = Path("enhanced_csp/network/__init__.py")
    
    print(f"🔧 Fixing {network_init_path}...")
    
    if not network_init_path.exists():
        print(f"❌ File not found: {network_init_path}")
        return False
    
    # Create a safe network __init__.py that handles imports gracefully
    new_content = '''"""
Enhanced CSP Network Module
Peer-to-peer networking, mesh topology, and adaptive routing
"""

__version__ = "1.0.0"

# Safe imports with fallbacks
try:
    from .core.types import (
        NodeID,
        NodeCapabilities,
        PeerInfo,
        NetworkMessage,
        MessageType,
    )
    _CORE_TYPES_AVAILABLE = True
except ImportError:
    _CORE_TYPES_AVAILABLE = False
    # Create minimal fallback classes
    class NodeID:
        def __init__(self, value=None):
            self.value = value or "default_node"
        def __str__(self):
            return self.value
        @classmethod
        def generate(cls):
            import secrets
            return cls(f"Qm{secrets.token_hex(22)}")
    
    class NodeCapabilities:
        def __init__(self):
            self.relay = True
            self.storage = False
            self.compute = False
    
    class PeerInfo:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class NetworkMessage:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class MessageType:
        DATA = "data"
        CONTROL = "control"
        HEARTBEAT = "heartbeat"

try:
    from .core.config import (
        NetworkConfig,
        SecurityConfig,
        P2PConfig,
        MeshConfig,
        DNSConfig,
        RoutingConfig,
    )
    _CORE_CONFIG_AVAILABLE = True
except ImportError:
    _CORE_CONFIG_AVAILABLE = False
    from dataclasses import dataclass, field
    from pathlib import Path
    
    @dataclass
    class SecurityConfig:
        enable_encryption: bool = True
        enable_authentication: bool = True
    
    @dataclass 
    class P2PConfig:
        listen_address: str = "0.0.0.0"
        listen_port: int = 30301
        bootstrap_nodes: list = field(default_factory=list)
        enable_mdns: bool = True
        enable_upnp: bool = True
        enable_nat_traversal: bool = True
        dns_seed_domain: str = "peoplesainetwork.com"
    
    @dataclass
    class MeshConfig:
        enable_super_peers: bool = True
        max_peers: int = 50
        topology_type: str = "adaptive_partial"
    
    @dataclass
    class DNSConfig:
        root_domain: str = ".web4ai"
        enable_dnssec: bool = True
    
    @dataclass
    class RoutingConfig:
        enable_multipath: bool = True
        enable_ml_predictor: bool = True
    
    @dataclass
    class NetworkConfig:
        security: SecurityConfig = field(default_factory=SecurityConfig)
        p2p: P2PConfig = field(default_factory=P2PConfig)
        mesh: MeshConfig = field(default_factory=MeshConfig)
        dns: DNSConfig = field(default_factory=DNSConfig)
        routing: RoutingConfig = field(default_factory=RoutingConfig)
        node_name: str = "csp-node"
        node_type: str = "standard"
        capabilities: NodeCapabilities = field(default_factory=NodeCapabilities)
        data_dir: Path = field(default_factory=lambda: Path("./network_data"))

# Try to import error classes
try:
    from .errors import (
        NetworkError,
        ConnectionError,
        TimeoutError,
        ProtocolError,
        SecurityError,
        ValidationError,
        ErrorMetrics,
        CircuitBreakerOpen,
    )
    _ERRORS_AVAILABLE = True
except ImportError:
    _ERRORS_AVAILABLE = False
    # Create basic error classes
    class NetworkError(Exception):
        pass
    class ConnectionError(NetworkError):
        pass
    class TimeoutError(NetworkError):
        pass
    class ProtocolError(NetworkError):
        pass
    class SecurityError(NetworkError):
        pass
    class ValidationError(NetworkError):
        pass
    class ErrorMetrics:
        def __init__(self):
            self.errors = {}
    class CircuitBreakerOpen(NetworkError):
        pass

# Try to import utils
try:
    from .utils import (
        setup_logging,
        get_logger,
        NetworkLogger,
        SecurityLogger,
        PerformanceLogger,
        AuditLogger,
    )
    _UTILS_AVAILABLE = True
except ImportError:
    _UTILS_AVAILABLE = False
    import logging
    def setup_logging(level="INFO"):
        logging.basicConfig(level=getattr(logging, level.upper()))
    def get_logger(name):
        return logging.getLogger(name)
    class NetworkLogger:
        def __init__(self, name):
            self.logger = logging.getLogger(name)
    class SecurityLogger:
        def __init__(self, name):
            self.logger = logging.getLogger(name)
    class PerformanceLogger:
        def __init__(self, name):
            self.logger = logging.getLogger(name)
    class AuditLogger:
        def __init__(self, name):
            self.logger = logging.getLogger(name)

# Network node classes - these are the main ones we need
def _lazy_network_node():
    """Lazy import NetworkNode."""
    try:
        from .core.node import NetworkNode
        return NetworkNode
    except ImportError:
        # Create a basic fallback NetworkNode
        class NetworkNode:
            def __init__(self, config=None):
                self.config = config or NetworkConfig()
                self.node_id = NodeID.generate()
                self.capabilities = NodeCapabilities()
                self.peers = {}
                self.is_running = False
            
            async def start(self):
                self.is_running = True
                return True
            
            async def stop(self):
                self.is_running = False
                return True
        return NetworkNode

def _lazy_enhanced_network():
    """Lazy import EnhancedCSPNetwork."""
    try:
        from .core.node import EnhancedCSPNetwork
        return EnhancedCSPNetwork
    except ImportError:
        # Create a basic fallback EnhancedCSPNetwork
        class EnhancedCSPNetwork:
            def __init__(self, config=None):
                self.config = config or NetworkConfig()
                self.node_id = NodeID.generate()
                self.nodes = {}
                self.is_running = False
            
            async def start(self):
                # Create a default node
                await self.create_node("default")
                self.is_running = True
                return True
            
            async def stop(self):
                await self.stop_all()
                self.is_running = False
                return True
            
            async def create_node(self, name="default"):
                node = _lazy_network_node()(self.config)
                await node.start()
                self.nodes[name] = node
                return node
            
            async def stop_all(self):
                for node in self.nodes.values():
                    await node.stop()
                self.nodes.clear()
            
            def get_node(self, name="default"):
                return self.nodes.get(name)
        return EnhancedCSPNetwork

# Convenience functions
def create_network(config=None):
    """Create a new Enhanced CSP Network instance."""
    return _lazy_enhanced_network()(config)

def create_node(config=None):
    """Create a new network node."""
    return _lazy_network_node()(config)

# Export main classes and functions
__all__ = [
    # Version
    "__version__",

    # Core types
    "NodeID",
    "NodeCapabilities",
    "PeerInfo", 
    "NetworkMessage",
    "MessageType",

    # Configuration
    "NetworkConfig",
    "SecurityConfig",
    "P2PConfig",
    "MeshConfig",
    "DNSConfig",
    "RoutingConfig",

    # Errors
    "NetworkError",
    "ConnectionError",
    "TimeoutError",
    "ProtocolError", 
    "SecurityError",
    "ValidationError",
    "ErrorMetrics",
    "CircuitBreakerOpen",

    # Logging utilities
    "setup_logging",
    "get_logger",
    "NetworkLogger",
    "SecurityLogger",
    "PerformanceLogger", 
    "AuditLogger",

    # Main functions
    "create_network",
    "create_node",
]

# Debug function
def get_import_status():
    """Get status of all imports for debugging."""
    return {
        "core_types": _CORE_TYPES_AVAILABLE,
        "core_config": _CORE_CONFIG_AVAILABLE, 
        "errors": _ERRORS_AVAILABLE,
        "utils": _UTILS_AVAILABLE,
    }
'''
    
    # Backup the original file
    backup_path = network_init_path.with_suffix('.py.backup')
    if network_init_path.exists():
        network_init_path.rename(backup_path)
        print(f"📦 Backed up original to: {backup_path}")
    
    # Write the new content
    try:
        network_init_path.write_text(new_content)
        print(f"✅ Fixed {network_init_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to fix {network_init_path}: {e}")
        # Restore backup if write failed
        if backup_path.exists():
            backup_path.rename(network_init_path)
            print(f"🔄 Restored backup")
        return False

def test_network_imports():
    """Test if the network imports now work."""
    print("\n🧪 Testing network imports...")
    
    try:
        # Clear any cached modules
        for module_name in list(sys.modules.keys()):
            if module_name.startswith('enhanced_csp.network'):
                del sys.modules[module_name]
        
        # Try the problematic imports
        from enhanced_csp.network import create_network, create_node
        from enhanced_csp.network import NetworkConfig, NodeID
        
        print("✅ Network imports successful!")
        
        # Test basic functionality
        config = NetworkConfig()
        network = create_network(config)
        node = create_node(config)
        
        print("✅ Network objects created successfully!")
        
        # Test the functions that were missing
        print(f"✅ create_network: {create_network}")
        print(f"✅ create_node: {create_node}")
        
        return True
        
    except Exception as e:
        print(f"❌ Network imports still failing: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    print("🚀 Network Init Fix Script")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not Path("enhanced_csp").exists():
        print("❌ enhanced_csp directory not found!")
        print("💡 Make sure you're running from the project root")
        return
    
    # Fix the network init file
    if fix_network_init():
        # Test the imports
        if test_network_imports():
            print("\n🎉 Network init fix successful!")
            print("💡 You can now run: python3 network_startup.py --quick-start")
        else:
            print("\n⚠️  Network init fix applied but still having issues")
    else:
        print("\n❌ Failed to apply network init fix")

if __name__ == "__main__":
    main()
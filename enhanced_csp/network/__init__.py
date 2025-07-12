# Here are all the missing __init__.py files you need to create:

# ==============================================================================
# FILE: enhanced_csp/network/__init__.py
# ==============================================================================
# enhanced_csp/network/__init__.py
"""
Enhanced CSP Network Module
Peer-to-peer networking, mesh topology, and adaptive routing
"""

__version__ = "1.0.0"

# Core network types and classes - lazy imports to avoid circular dependencies
def _lazy_import():
    try:
        from .core.types import (
            NodeID,
            NodeCapabilities,
            PeerInfo,
            NetworkMessage,
            MessageType,
        )
        
        from .core.config import (
            NetworkConfig,
            SecurityConfig,
            P2PConfig,
            MeshConfig,
            DNSConfig,
            RoutingConfig,
            PQCConfig,
        )
        
        return {
            'NodeID': NodeID,
            'NodeCapabilities': NodeCapabilities,
            'PeerInfo': PeerInfo,
            'NetworkMessage': NetworkMessage,
            'MessageType': MessageType,
            'NetworkConfig': NetworkConfig,
            'SecurityConfig': SecurityConfig,
            'P2PConfig': P2PConfig,
            'MeshConfig': MeshConfig,
            'DNSConfig': DNSConfig,
            'RoutingConfig': RoutingConfig,
            'PQCConfig': PQCConfig,
        }
    except ImportError:
        return {}

# Module globals for lazy loading
_IMPORTS = None

def __getattr__(name):
    """Lazy import mechanism."""
    global _IMPORTS
    if _IMPORTS is None:
        _IMPORTS = _lazy_import()
    
    if name in _IMPORTS:
        return _IMPORTS[name]
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

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
    "PQCConfig",
]

# ==============================================================================
# FILE: enhanced_csp/network/core/__init__.py  
# ==============================================================================
# enhanced_csp/network/core/__init__.py
"""Core network components."""

try:
    from .config import (
        NetworkConfig,
        SecurityConfig,
        P2PConfig,
        MeshConfig,
        DNSConfig,
        RoutingConfig,
        PQCConfig,
    )
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

try:
    from .types import (
        NodeID, NodeCapabilities, MessageType, PeerInfo, NetworkMessage
    )
    TYPES_AVAILABLE = True
except ImportError:
    TYPES_AVAILABLE = False

# Export available components
__all__ = []

if CONFIG_AVAILABLE:
    __all__.extend([
        'NetworkConfig',
        'SecurityConfig', 
        'P2PConfig',
        'MeshConfig',
        'DNSConfig',
        'RoutingConfig',
        'PQCConfig',
    ])

if TYPES_AVAILABLE:
    __all__.extend([
        'NodeID',
        'NodeCapabilities',
        'MessageType',
        'PeerInfo', 
        'NetworkMessage',
    ])

# ==============================================================================
# FILE: enhanced_csp/network/security/__init__.py
# ==============================================================================
# enhanced_csp/network/security/__init__.py
"""Security components for Enhanced CSP Network."""

try:
    from .security_hardening import (
        SecurityOrchestrator,
        MessageValidator,
        SecurityInputValidator,
        SecureTLSConfig,
        RateLimiter,
        SecureImporter,
        validate_ip_address,
        validate_port_number,
    )
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False

__all__ = []

if SECURITY_AVAILABLE:
    __all__.extend([
        'SecurityOrchestrator',
        'MessageValidator', 
        'SecurityInputValidator',
        'SecureTLSConfig',
        'RateLimiter',
        'SecureImporter',
        'validate_ip_address',
        'validate_port_number',
    ])

# ==============================================================================
# FILE: enhanced_csp/network/p2p/__init__.py
# ==============================================================================  
# enhanced_csp/network/p2p/__init__.py
"""Peer-to-peer networking components."""

__all__ = []

# ==============================================================================
# FILE: enhanced_csp/network/mesh/__init__.py
# ==============================================================================
# enhanced_csp/network/mesh/__init__.py
"""Mesh networking components."""

__all__ = []

# ==============================================================================
# FILE: enhanced_csp/network/dns/__init__.py
# ==============================================================================
# enhanced_csp/network/dns/__init__.py
"""DNS overlay components."""

__all__ = []

# ==============================================================================
# FILE: enhanced_csp/network/routing/__init__.py
# ==============================================================================
# enhanced_csp/network/routing/__init__.py  
"""Routing components."""

__all__ = []

# ==============================================================================
# INSTRUCTIONS FOR CREATING FILES
# ==============================================================================

"""
To fix the import issues, you need to create these __init__.py files:

1. Create enhanced_csp/__init__.py (already provided above)
2. Create enhanced_csp/network/__init__.py  
3. Create enhanced_csp/network/core/__init__.py
4. Create enhanced_csp/network/security/__init__.py
5. Create enhanced_csp/network/utils/__init__.py (already exists)
6. Create enhanced_csp/network/p2p/__init__.py
7. Create enhanced_csp/network/mesh/__init__.py
8. Create enhanced_csp/network/dns/__init__.py
9. Create enhanced_csp/network/routing/__init__.py

Copy each section above into the corresponding file path.

Alternatively, run this command from your project root:

# Create all missing __init__.py files at once:
python -c "
import os
from pathlib import Path

dirs = [
    'enhanced_csp',
    'enhanced_csp/network', 
    'enhanced_csp/network/core',
    'enhanced_csp/network/security',
    'enhanced_csp/network/p2p',
    'enhanced_csp/network/mesh', 
    'enhanced_csp/network/dns',
    'enhanced_csp/network/routing'
]

for dir_path in dirs:
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    init_file = Path(dir_path) / '__init__.py'
    if not init_file.exists():
        init_file.write_text(f'# {Path(dir_path).name} package\\n')
        print(f'Created {init_file}')
"

Then run: python enhanced_csp/network/test_optimizations.py
"""
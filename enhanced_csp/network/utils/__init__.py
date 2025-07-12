"""Network utility functions and helpers.

Provides common utilities for networking operations, data processing,
and system integration.
"""

import logging
from typing import Any, Optional, Union

# Import functions from validation module
try:
    from .validation import (
        validate_ip_address,
        validate_port_number, 
        validate_node_id,
        validate_message_size,
        sanitize_string_input,
        validate_input,
        PeerRateLimiter,
        MAX_MESSAGE_SIZE
    )
    _validation_available = True
except ImportError:
    _validation_available = False
    # Fallback implementations
    def validate_message_size(data, max_size=1024*1024):
        """Fallback message size validator."""
        import json
        if isinstance(data, (bytes, bytearray)):
            size = len(data)
        elif isinstance(data, str):
            size = len(data.encode())
        else:
            try:
                serialized = json.dumps(data).encode()
                size = len(serialized)
            except Exception:
                size = 0
        if size > max_size:
            raise ValueError(f"Message too large: {size} bytes")
    
    def validate_ip_address(ip_str):
        """Fallback IP validator."""
        import ipaddress
        try:
            ipaddress.ip_address(ip_str)
            return ip_str
        except ValueError as e:
            raise ValueError(f"Invalid IP address: {ip_str}") from e
    
    def validate_port_number(port):
        """Fallback port validator."""
        try:
            port_int = int(port)
            if not (1 <= port_int <= 65535):
                raise ValueError(f"Invalid port number: {port}")
            return port_int
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid port number: {port}") from e

# Import logging utilities
try:
    from .structured_logging import (
        get_logger,
        setup_logging,
        NetworkLogger,
        SecurityLogger,
        PerformanceLogger,
        AuditLogger
    )
    _logging_available = True
except ImportError:
    _logging_available = False
    # Fallback implementations
    def get_logger(name: str) -> logging.Logger:
        """Get a logger with the specified name (fallback)."""
        return logging.getLogger(name)
    
    def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
        """Setup basic logging (fallback)."""
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        handler = logging.StreamHandler()
        handler.setLevel(numeric_level)
        handler.setFormatter(formatter)
        
        root_logger = logging.getLogger()
        root_logger.setLevel(numeric_level)
        root_logger.handlers.clear()
        root_logger.addHandler(handler)

# Try to import from the parent utils.py file
try:
    import sys
    import os
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    utils_file = os.path.join(parent_dir, 'utils.py')
    if os.path.exists(utils_file):
        # Add parent directory to path temporarily
        sys.path.insert(0, parent_dir)
        try:
            from utils import (
                format_bytes,
                format_duration,
                get_local_ip,
                is_port_available
            )
            _parent_utils_available = True
        except ImportError:
            _parent_utils_available = False
        finally:
            # Remove from path
            if parent_dir in sys.path:
                sys.path.remove(parent_dir)
    else:
        _parent_utils_available = False
except Exception:
    _parent_utils_available = False
    
    # Fallback utility functions
    def format_bytes(bytes_value: int) -> str:
        """Format bytes into human readable string."""
        if bytes_value == 0:
            return "0 B"
        units = ["B", "KB", "MB", "GB", "TB"]
        unit_index = 0
        size = float(bytes_value)
        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1
        return f"{size:.1f} {units[unit_index]}"
    
    def get_local_ip() -> str:
        """Get local IP address."""
        return "127.0.0.1"

# Import other utility modules if available
try:
    from .helpers import *
except ImportError:
    pass

try:
    from .validators import *
except ImportError:
    pass

try:
    from .converters import *
except ImportError:
    pass

__all__ = [
    # Core validation functions
    "validate_ip_address",
    "validate_port_number",
    "validate_message_size",
    
    # Additional validation functions (if available)
    "validate_node_id",
    "sanitize_string_input", 
    "validate_input",
    "PeerRateLimiter",
    "MAX_MESSAGE_SIZE",
    
    # Logging functions
    "get_logger",
    "setup_logging",
    
    # Structured logging (if available)
    "NetworkLogger",
    "SecurityLogger", 
    "PerformanceLogger",
    "AuditLogger",
    
    # Utility functions (if available)
    "format_bytes",
    "format_duration",
    "get_local_ip",
    "is_port_available",
    
    # Module groups
    "helpers",
    "validators", 
    "converters",
]

# Module metadata
__version__ = "1.0.0"
__description__ = "Enhanced CSP Network Utilities"

# Debug function to check what's available
def get_import_status():
    """Return status of all imports for debugging."""
    return {
        "validation_module": _validation_available,
        "logging_module": _logging_available,
        "parent_utils": _parent_utils_available,
        "available_functions": [name for name in __all__ if name in globals()]
    }
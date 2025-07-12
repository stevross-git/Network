# enhanced_csp/network/utils/__init__.py
"""Utility functions and classes for Enhanced CSP Network."""

import logging
import time
import ipaddress
from typing import Any, Optional, Union
from pathlib import Path

# Import validation functions with fallbacks
try:
    from .validation import validate_ip_address, validate_port_number
except ImportError:
    def validate_ip_address(ip_str: str) -> str:
        """Validate IP address (fallback implementation)."""
        try:
            ip = ipaddress.ip_address(ip_str.strip())
            return str(ip)
        except ValueError as e:
            raise ValueError(f"Invalid IP address: {ip_str}") from e
    
    def validate_port_number(port: Union[int, str]) -> int:
        """Validate port number (fallback implementation)."""
        try:
            port_int = int(port)
            if not (1 <= port_int <= 65535):
                raise ValueError(f"Port out of range: {port_int}")
            return port_int
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid port: {port}") from e

# Import logging utilities with fallbacks
try:
    from .structured_logging import (
        setup_logging, get_logger, NetworkLogger, SecurityLogger,
        PerformanceLogger, AuditLogger
    )
except ImportError:
    def setup_logging(level: int = logging.INFO, 
                     log_file: Optional[Path] = None) -> None:
        """Setup basic logging (fallback implementation)."""
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        if log_file:
            logging.basicConfig(
                level=level,
                format=format_str,
                filename=str(log_file),
                filemode='a'
            )
        else:
            logging.basicConfig(level=level, format=format_str)
    
    def get_logger(name: str) -> logging.Logger:
        """Get a logger instance (fallback implementation)."""
        return logging.getLogger(name)
    
    class NetworkLogger:
        """Network logger (fallback implementation)."""
        def __init__(self, name: str = "network"):
            self.logger = logging.getLogger(name)
        
        def info(self, message: str, **kwargs):
            self.logger.info(message, extra=kwargs)
        
        def warning(self, message: str, **kwargs):
            self.logger.warning(message, extra=kwargs)
        
        def error(self, message: str, **kwargs):
            self.logger.error(message, extra=kwargs)
    
    class SecurityLogger(NetworkLogger):
        """Security logger (fallback implementation)."""
        def __init__(self):
            super().__init__("security")
    
    class PerformanceLogger(NetworkLogger):
        """Performance logger (fallback implementation)."""
        def __init__(self):
            super().__init__("performance")
    
    class AuditLogger(NetworkLogger):
        """Audit logger (fallback implementation)."""
        def __init__(self):
            super().__init__("audit")

# Import other utilities with fallbacks
try:
    from .rate_limit import RateLimiter
except ImportError:
    import asyncio
    import time
    
    class RateLimiter:
        """Rate limiter (fallback implementation)."""
        
        def __init__(self, max_tokens: int = 100, refill_rate: float = 10.0):
            self.max_tokens = max_tokens
            self.refill_rate = refill_rate
            self.tokens = max_tokens
            self.last_refill = time.time()
            self._lock = asyncio.Lock()
        
        async def acquire(self, tokens: int = 1) -> bool:
            """Acquire tokens from the bucket."""
            async with self._lock:
                now = time.time()
                time_passed = now - self.last_refill
                
                # Refill tokens
                self.tokens = min(
                    self.max_tokens,
                    self.tokens + time_passed * self.refill_rate
                )
                self.last_refill = now
                
                # Check if we have enough tokens
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True
                
                return False

try:
    from .retry import retry_with_backoff
except ImportError:
    import asyncio
    import random
    
    async def retry_with_backoff(func, max_retries: int = 3, 
                               base_delay: float = 1.0, max_delay: float = 60.0):
        """Retry function with exponential backoff (fallback implementation)."""
        for attempt in range(max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func()
                else:
                    return func()
            except Exception as e:
                if attempt == max_retries:
                    raise e
                
                delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                await asyncio.sleep(delay)

try:
    from .secure_random import secure_bytes, secure_string
except ImportError:
    import secrets
    import string
    
    def secure_bytes(length: int) -> bytes:
        """Generate secure random bytes (fallback implementation)."""
        return secrets.token_bytes(length)
    
    def secure_string(length: int) -> str:
        """Generate secure random string (fallback implementation)."""
        alphabet = string.ascii_letters + string.digits
        return ''.join(secrets.choice(alphabet) for _ in range(length))

# Timer utility class
class Timer:
    """Simple timer utility for performance measurement."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start the timer."""
        self.start_time = time.perf_counter()
        self.end_time = None
    
    def stop(self):
        """Stop the timer."""
        if self.start_time is None:
            raise ValueError("Timer not started")
        self.end_time = time.perf_counter()
    
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            raise ValueError("Timer not started")
        
        end_time = self.end_time or time.perf_counter()
        return end_time - self.start_time
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

# Utility functions for formatting
def format_bytes(num_bytes: int) -> str:
    """Format bytes in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"

def format_duration(seconds: float) -> str:
    """Format duration in human readable format."""
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def format_rate(count: int, duration: float) -> str:
    """Format rate (count per second)."""
    if duration <= 0:
        return "0.0/s"
    rate = count / duration
    if rate < 1:
        return f"{rate:.3f}/s"
    elif rate < 1000:
        return f"{rate:.1f}/s"
    else:
        return f"{rate/1000:.1f}k/s"

# Configuration helpers
def load_config_file(config_path: Path) -> dict:
    """Load configuration from file with format detection."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    if config_path.suffix.lower() == '.json':
        import json
        with config_path.open('r') as f:
            return json.load(f)
    elif config_path.suffix.lower() in ['.yml', '.yaml']:
        try:
            import yaml
            with config_path.open('r') as f:
                return yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML required for YAML configuration files")
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")

def save_config_file(config: dict, config_path: Path) -> None:
    """Save configuration to file with format detection."""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    if config_path.suffix.lower() == '.json':
        import json
        with config_path.open('w') as f:
            json.dump(config, f, indent=2)
    elif config_path.suffix.lower() in ['.yml', '.yaml']:
        try:
            import yaml
            with config_path.open('w') as f:
                yaml.safe_dump(config, f, default_flow_style=False)
        except ImportError:
            raise ImportError("PyYAML required for YAML configuration files")
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")

# Export all utilities
__all__ = [
    # Validation
    "validate_ip_address",
    "validate_port_number",
    
    # Logging
    "setup_logging",
    "get_logger",
    "NetworkLogger",
    "SecurityLogger", 
    "PerformanceLogger",
    "AuditLogger",
    
    # Rate limiting
    "RateLimiter",
    
    # Retry logic
    "retry_with_backoff",
    
    # Secure random
    "secure_bytes",
    "secure_string",
    
    # Timer and formatting
    "Timer",
    "format_bytes",
    "format_duration",
    "format_rate",
    
    # Configuration
    "load_config_file",
    "save_config_file",
]

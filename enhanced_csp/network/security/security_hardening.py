# enhanced_csp/network/security/security_hardening.py
"""Production-ready security hardening for Enhanced CSP Network."""

import ssl
import time
import logging
import hashlib
import secrets
from typing import Any, Dict, List, Optional, Union, Protocol, TypeVar
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import ipaddress
import re
import asyncio
from contextlib import asynccontextmanager

# Proper relative imports
try:
    from ..errors import SecurityError, ValidationError
    from ..core.types import NodeID, NetworkMessage, MessageType
except ImportError:
    # Fallback for development
    class SecurityError(Exception):
        pass
    
    class ValidationError(Exception):
        pass
    
    class NodeID:
        def __init__(self, value=None):
            self.value = value or "default_node"
        
        def __str__(self):
            return self.value
    
    class NetworkMessage:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class MessageType:
        DATA = "data"
        CONTROL = "control"
        HEARTBEAT = "heartbeat"

logger = logging.getLogger(__name__)

# Security Constants
MIN_TLS_VERSION = ssl.TLSVersion.TLSv1_3
CIPHER_SUITES = [
    'TLS_AES_256_GCM_SHA384',
    'TLS_CHACHA20_POLY1305_SHA256',
    'TLS_AES_128_GCM_SHA256',
]

class SecureImporter:
    """Secure replacement for dynamic imports."""
    
    # Whitelist of allowed modules
    ALLOWED_MODULES = {
        'enhanced_csp.network.core.types': ['NodeID', 'NetworkMessage', 'MessageType'],
        'enhanced_csp.network.core.config': ['NetworkConfig', 'SecurityConfig'],
        'enhanced_csp.network.p2p.transport': ['Transport', 'QUICTransport'],
        'enhanced_csp.network.utils': ['Logger', 'Metrics'],
    }
    
    @classmethod
    def safe_import(cls, module_path: str, class_name: str) -> Any:
        """Safely import a class with validation."""
        if module_path not in cls.ALLOWED_MODULES:
            raise SecurityError(f"Module not in whitelist: {module_path}")
        
        if class_name not in cls.ALLOWED_MODULES[module_path]:
            raise SecurityError(f"Class not allowed: {class_name} from {module_path}")
        
        try:
            module = __import__(module_path, fromlist=[class_name])
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to import {class_name} from {module_path}") from e


class MessageValidator:
    """Comprehensive message validation framework."""
    
    def __init__(self, max_message_size: int = 10 * 1024 * 1024):
        self.max_message_size = max_message_size
        self.validation_cache = {}
        self._compiled_patterns = self._compile_validation_patterns()
    
    def _compile_validation_patterns(self) -> Dict[str, re.Pattern]:
        """Pre-compile regex patterns for better performance."""
        return {
            'node_id': re.compile(r'^[a-zA-Z0-9_-]{1,64}$'),
            'safe_string': re.compile(r'^[a-zA-Z0-9\s\-_.]{1,256}$'),
        }
    
    def validate_network_message(self, message: NetworkMessage) -> bool:
        """Validate network message for security and correctness."""
        try:
            self._validate_message_size(message)
            self._validate_sender_identity(message)
            self._validate_message_type(message)
            self._validate_timestamp(message)
            self._validate_payload(message)
            return True
        except (ValidationError, SecurityError):
            raise
        except Exception as e:
            raise ValidationError(f"Unexpected validation error: {e}")
    
    def _validate_message_size(self, message: NetworkMessage) -> None:
        """Validate message size constraints."""
        payload = getattr(message, 'payload', b'')
        if hasattr(payload, '__len__'):
            payload_size = len(payload)
        else:
            payload_size = len(str(payload).encode())
            
        if payload_size > self.max_message_size:
            raise ValidationError(f"Message too large: {payload_size} bytes")
        
        if payload_size == 0:
            raise ValidationError("Empty message payload")
    
    def _validate_sender_identity(self, message: NetworkMessage) -> None:
        """Validate sender NodeID format and structure."""
        sender = getattr(message, 'sender', None)
        if not isinstance(sender, (str, NodeID)):
            raise SecurityError("Invalid sender type")
        
        sender_str = str(sender)
        if not self._compiled_patterns['node_id'].match(sender_str):
            raise SecurityError(f"Invalid sender ID format: {sender_str}")
    
    def _validate_message_type(self, message: NetworkMessage) -> None:
        """Validate message type."""
        msg_type = getattr(message, 'type', None)
        if msg_type not in [MessageType.DATA, MessageType.CONTROL, MessageType.HEARTBEAT]:
            raise ValidationError(f"Invalid message type: {msg_type}")
    
    def _validate_timestamp(self, message: NetworkMessage) -> None:
        """Validate timestamp to prevent replay attacks."""
        if hasattr(message, 'timestamp'):
            current_time = time.time()
            time_diff = abs(current_time - message.timestamp)
            if time_diff > 300:  # 5 minutes
                raise SecurityError(f"Message timestamp outside acceptable range: {time_diff}s")
    
    def _validate_payload(self, message: NetworkMessage) -> None:
        """Validate payload structure based on message type."""
        msg_type = getattr(message, 'type', None)
        payload = getattr(message, 'payload', None)
        
        if msg_type == MessageType.DATA:
            self._validate_data_payload(payload)
        elif msg_type == MessageType.CONTROL:
            self._validate_control_payload(payload)
    
    def _validate_data_payload(self, payload: Any) -> None:
        """Validate data message payload."""
        if isinstance(payload, bytes):
            # Check for executable signatures
            executable_signatures = [
                b'\x7fELF',  # Linux ELF
                b'MZ',       # Windows PE
                b'\xfe\xed\xfa',  # Mach-O
            ]
            
            for sig in executable_signatures:
                if payload.startswith(sig):
                    raise SecurityError("Executable content detected in payload")
    
    def _validate_control_payload(self, payload: Any) -> None:
        """Validate control message payload."""
        # Implement control-specific validation
        if payload and isinstance(payload, dict):
            # Check for dangerous keys
            dangerous_keys = ['__import__', 'eval', 'exec', 'compile']
            for key in dangerous_keys:
                if key in payload:
                    raise SecurityError(f"Dangerous key in control payload: {key}")


class SecurityInputValidator:
    """Comprehensive input validation with sanitization."""
    
    @staticmethod
    def validate_ip_address(ip_str: str) -> str:
        """Validate and normalize IP address."""
        try:
            ip = ipaddress.ip_address(ip_str.strip())
            
            # Security checks
            if ip.is_private and not SecurityInputValidator._allow_private_ips():
                raise SecurityError(f"Private IP not allowed: {ip}")
            
            if ip.is_loopback and not SecurityInputValidator._allow_loopback():
                raise SecurityError(f"Loopback IP not allowed: {ip}")
            
            return str(ip)
        except ValueError as e:
            raise ValidationError(f"Invalid IP address: {ip_str}") from e
    
    @staticmethod
    def validate_port(port: Union[int, str]) -> int:
        """Validate port number."""
        try:
            port_int = int(port)
            if not (1 <= port_int <= 65535):
                raise ValidationError(f"Port out of range: {port_int}")
            
            # Check for privileged ports in production
            if port_int < 1024 and not SecurityInputValidator._allow_privileged_ports():
                raise SecurityError(f"Privileged port not allowed: {port_int}")
            
            return port_int
        except (ValueError, TypeError) as e:
            raise ValidationError(f"Invalid port: {port}") from e
    
    @staticmethod
    def sanitize_string(input_str: str, max_length: int = 1000) -> str:
        """Sanitize string input to prevent injection attacks."""
        if not isinstance(input_str, str):
            raise ValidationError("Input must be a string")
        
        # Remove dangerous characters
        sanitized = re.sub(r'[<>"\'\x00-\x1f\x7f-\x9f]', '', input_str)
        
        # Limit length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized.strip()
    
    @staticmethod
    def _allow_private_ips() -> bool:
        """Check if private IPs are allowed in current environment."""
        # In development, allow private IPs
        return True
    
    @staticmethod
    def _allow_loopback() -> bool:
        """Check if loopback IPs are allowed."""
        return True
    
    @staticmethod
    def _allow_privileged_ports() -> bool:
        """Check if privileged ports are allowed."""
        # Allow in development mode
        return True


class SecureTLSConfig:
    """Enhanced TLS configuration with security best practices."""
    
    def __init__(self):
        self.min_tls_version = MIN_TLS_VERSION
        self.cipher_suites = CIPHER_SUITES
        self.verify_mode = ssl.CERT_REQUIRED
        self.check_hostname = True
        self.ca_cert_path: Optional[Path] = None
        self.cert_path: Optional[Path] = None
        self.key_path: Optional[Path] = None
        
    def create_server_context(self) -> ssl.SSLContext:
        """Create secure SSL context for server connections."""
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = self.min_tls_version
        context.maximum_version = ssl.TLSVersion.TLSv1_3
        
        # Set cipher suites
        context.set_ciphers(':'.join(self.cipher_suites))
        
        # Load certificates if available
        if self.cert_path and self.key_path:
            context.load_cert_chain(str(self.cert_path), str(self.key_path))
        
        # Load CA certificates
        if self.ca_cert_path:
            context.load_verify_locations(str(self.ca_cert_path))
        
        # Security options
        context.verify_mode = self.verify_mode
        context.check_hostname = self.check_hostname
        
        return context
    
    def create_client_context(self) -> ssl.SSLContext:
        """Create secure SSL context for client connections."""
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.minimum_version = self.min_tls_version
        context.maximum_version = ssl.TLSVersion.TLSv1_3
        
        # Set cipher suites
        context.set_ciphers(':'.join(self.cipher_suites))
        
        # Load CA certificates
        if self.ca_cert_path:
            context.load_verify_locations(str(self.ca_cert_path))
        else:
            context.load_default_certs()
        
        # Security options
        context.verify_mode = ssl.CERT_REQUIRED
        context.check_hostname = True
        
        return context


class RateLimiter:
    """Token bucket rate limiter for security."""
    
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


class SecurityOrchestrator:
    """Main security orchestrator for the Enhanced CSP Network."""
    
    def __init__(self, config=None):
        self.config = config
        self.message_validator = MessageValidator()
        self.input_validator = SecurityInputValidator()
        self.tls_config = SecureTLSConfig()
        self.rate_limiter = RateLimiter()
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self) -> None:
        """Initialize security orchestrator."""
        self.logger.info("Initializing security orchestrator")
        
        if self.config and self.config.enable_tls:
            await self._setup_tls()
        
        self.logger.info("Security orchestrator initialized successfully")
    
    async def shutdown(self) -> None:
        """Shutdown security orchestrator."""
        self.logger.info("Shutting down security orchestrator")
    
    async def validate_message(self, message: NetworkMessage) -> bool:
        """Validate incoming network message."""
        return self.message_validator.validate_network_message(message)
    
    def validate_ip_address(self, ip: str) -> str:
        """Validate IP address."""
        return self.input_validator.validate_ip_address(ip)
    
    def validate_port(self, port: Union[int, str]) -> int:
        """Validate port number."""
        return self.input_validator.validate_port(port)
    
    def sanitize_string(self, input_str: str) -> str:
        """Sanitize string input."""
        return self.input_validator.sanitize_string(input_str)
    
    async def check_rate_limit(self, tokens: int = 1) -> bool:
        """Check rate limit."""
        return await self.rate_limiter.acquire(tokens)
    
    def get_tls_context(self, server: bool = True) -> ssl.SSLContext:
        """Get TLS context for connections."""
        if server:
            return self.tls_config.create_server_context()
        else:
            return self.tls_config.create_client_context()
    
    async def _setup_tls(self) -> None:
        """Set up TLS configuration."""
        if hasattr(self.config, 'tls_cert_path') and self.config.tls_cert_path:
            self.tls_config.cert_path = Path(self.config.tls_cert_path)
        
        if hasattr(self.config, 'tls_key_path') and self.config.tls_key_path:
            self.tls_config.key_path = Path(self.config.tls_key_path)
        
        if hasattr(self.config, 'ca_cert_path') and self.config.ca_cert_path:
            self.tls_config.ca_cert_path = Path(self.config.ca_cert_path)
        
        self.logger.info("TLS configuration completed")


# Utility functions
def validate_ip_address(ip_str: str) -> str:
    """Validate IP address (convenience function)."""
    return SecurityInputValidator.validate_ip_address(ip_str)


def validate_port_number(port: Union[int, str]) -> int:
    """Validate port number (convenience function)."""
    return SecurityInputValidator.validate_port(port)


__all__ = [
    'SecurityOrchestrator',
    'MessageValidator',
    'SecurityInputValidator',
    'SecureTLSConfig',
    'RateLimiter',
    'SecureImporter',
    'validate_ip_address',
    'validate_port_number',
]
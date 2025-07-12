# enhanced_csp/network/security/security_hardening.py
"""
Production-ready security hardening for Enhanced CSP Network.
Addresses critical security vulnerabilities identified in code review.
"""

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

from ..errors import SecurityError, ValidationError
from ..core.types import NodeID, NetworkMessage, MessageType

logger = logging.getLogger(__name__)

# Type safety enhancements
T = TypeVar('T')

class Serializable(Protocol):
    """Protocol for serializable objects."""
    def serialize(self) -> bytes: ...
    @classmethod
    def deserialize(cls, data: bytes) -> 'Serializable': ...

# Security constants
MAX_MESSAGE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_BATCH_SIZE = 100
MIN_TLS_VERSION = ssl.TLSVersion.TLSv1_3
CIPHER_SUITES = [
    "TLS_AES_256_GCM_SHA384",
    "TLS_CHACHA20_POLY1305_SHA256",
    "TLS_AES_128_GCM_SHA256"
]
RATE_LIMIT_WINDOW = 60  # seconds
MAX_REQUESTS_PER_WINDOW = 1000

class MessageValidator:
    """Comprehensive message validation framework."""
    
    def __init__(self, max_message_size: int = MAX_MESSAGE_SIZE):
        self.max_message_size = max_message_size
        self.validation_cache = {}
        
    def validate_network_message(self, message: NetworkMessage) -> bool:
        """
        Validate network message for security and correctness.
        
        Args:
            message: NetworkMessage to validate
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If message is invalid
            SecurityError: If message poses security risk
        """
        # Validate message size
        if len(message.payload) > self.max_message_size:
            raise ValidationError(f"Message too large: {len(message.payload)} bytes")
        
        # Validate sender identity
        if not self._validate_node_id(message.sender):
            raise SecurityError(f"Invalid sender ID: {message.sender}")
        
        # Validate message type
        if not isinstance(message.type, MessageType):
            raise ValidationError(f"Invalid message type: {message.type}")
        
        # Validate timestamp (prevent replay attacks)
        if hasattr(message, 'timestamp'):
            current_time = time.time()
            if abs(current_time - message.timestamp) > 300:  # 5 minutes
                raise SecurityError("Message timestamp outside acceptable range")
        
        # Validate payload structure
        self._validate_payload(message.payload, message.type)
        
        return True
    
    def _validate_node_id(self, node_id: NodeID) -> bool:
        """Validate NodeID format and structure."""
        if not isinstance(node_id, (str, NodeID)):
            return False
        
        node_str = str(node_id)
        # Basic format validation (adjust based on your NodeID format)
        if len(node_str) < 20 or len(node_str) > 128:
            return False
        
        # Check for malicious patterns
        if any(char in node_str for char in ['<', '>', '"', "'"]):
            return False
        
        return True
    
    def _validate_payload(self, payload: bytes, message_type: MessageType) -> None:
        """Validate message payload based on type."""
        if not payload and message_type != MessageType.HEARTBEAT:
            raise ValidationError("Empty payload for non-heartbeat message")
        
        # Type-specific validation
        if message_type == MessageType.DATA:
            # Check for binary content safety
            try:
                # Ensure payload is valid UTF-8 or safe binary
                if len(payload) > 0:
                    # Test for text content
                    try:
                        payload.decode('utf-8', errors='strict')
                    except UnicodeDecodeError:
                        # Binary content - additional checks
                        self._validate_binary_payload(payload)
            except Exception as e:
                raise ValidationError(f"Payload validation failed: {e}")
    
    def _validate_binary_payload(self, payload: bytes) -> None:
        """Additional validation for binary payloads."""
        # Check for executable signatures
        executable_signatures = [
            b'\x7fELF',  # Linux ELF
            b'MZ',       # Windows PE
            b'\xfe\xed\xfa',  # Mach-O
        ]
        
        for sig in executable_signatures:
            if payload.startswith(sig):
                raise SecurityError("Executable content detected in payload")


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
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        
        # Set minimum TLS version
        context.minimum_version = self.min_tls_version
        context.maximum_version = ssl.TLSVersion.TLSv1_3
        
        # Configure cipher suites
        context.set_ciphers(':'.join(self.cipher_suites))
        
        # Enhanced security settings
        context.verify_mode = ssl.CERT_NONE  # Will be set to REQUIRED after cert setup
        context.check_hostname = False  # Will be enabled for client connections
        
        # Disable compression (CRIME attack prevention)
        context.options |= ssl.OP_NO_COMPRESSION
        
        # Disable session tickets (privacy)
        context.options |= ssl.OP_NO_TICKET
        
        # Load certificates if provided
        if self.cert_path and self.key_path:
            context.load_cert_chain(str(self.cert_path), str(self.key_path))
            context.verify_mode = ssl.CERT_REQUIRED
        
        # Load CA certificates
        if self.ca_cert_path:
            context.load_verify_locations(str(self.ca_cert_path))
        
        return context
    
    def create_client_context(self) -> ssl.SSLContext:
        """Create secure SSL context for client connections."""
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        
        # Set minimum TLS version
        context.minimum_version = self.min_tls_version
        context.maximum_version = ssl.TLSVersion.TLSv1_3
        
        # Configure cipher suites
        context.set_ciphers(':'.join(self.cipher_suites))
        
        # Enhanced security settings
        context.verify_mode = self.verify_mode
        context.check_hostname = self.check_hostname
        
        # Disable compression and session tickets
        context.options |= ssl.OP_NO_COMPRESSION | ssl.OP_NO_TICKET
        
        # Load CA certificates
        if self.ca_cert_path:
            context.load_verify_locations(str(self.ca_cert_path))
        
        return context
    
    def validate_certificate(self, cert_path: Path) -> bool:
        """Validate TLS certificate before use."""
        try:
            import cryptography.x509
            from cryptography.hazmat.backends import default_backend
            
            with open(cert_path, 'rb') as f:
                cert_data = f.read()
            
            cert = cryptography.x509.load_pem_x509_certificate(cert_data, default_backend())
            
            # Check expiration
            current_time = time.time()
            if cert.not_valid_after.timestamp() <= current_time:
                logger.error("Certificate has expired")
                return False
            
            # Check validity start time
            if cert.not_valid_before.timestamp() > current_time:
                logger.error("Certificate not yet valid")
                return False
            
            # Additional checks can be added here
            return True
            
        except Exception as e:
            logger.error(f"Certificate validation failed: {e}")
            return False


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
            raise ValidationError("Input must be string")
        
        # Length check
        if len(input_str) > max_length:
            raise ValidationError(f"String too long: {len(input_str)} > {max_length}")
        
        # Remove dangerous characters
        dangerous_chars = ['<', '>', '"', "'", '&', '\0', '\r', '\n']
        sanitized = input_str
        
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        # Remove control characters
        sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', sanitized)
        
        return sanitized.strip()
    
    @staticmethod
    def validate_filename(filename: str) -> str:
        """Validate filename for path traversal attacks."""
        if not filename or not isinstance(filename, str):
            raise ValidationError("Invalid filename")
        
        # Check for path traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            raise SecurityError("Path traversal attempt detected")
        
        # Check for dangerous filenames
        dangerous_names = ['con', 'prn', 'aux', 'nul', 'com1', 'lpt1']
        if filename.lower() in dangerous_names:
            raise SecurityError(f"Dangerous filename: {filename}")
        
        return SecurityInputValidator.sanitize_string(filename, 255)
    
    @staticmethod
    def _allow_private_ips() -> bool:
        """Check if private IPs are allowed (configuration dependent)."""
        # This would be configurable in production
        return True
    
    @staticmethod
    def _allow_loopback() -> bool:
        """Check if loopback IPs are allowed."""
        return True
    
    @staticmethod
    def _allow_privileged_ports() -> bool:
        """Check if privileged ports are allowed."""
        # This would check for root privileges or capabilities
        return True


class RateLimiter:
    """Token bucket rate limiter for DOS protection."""
    
    def __init__(self, max_requests: int = MAX_REQUESTS_PER_WINDOW, 
                 window_seconds: int = RATE_LIMIT_WINDOW):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.buckets: Dict[str, Dict[str, Any]] = {}
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
    
    async def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed for given identifier."""
        current_time = time.time()
        
        # Periodic cleanup
        if current_time - self.last_cleanup > self.cleanup_interval:
            await self._cleanup_expired_buckets(current_time)
        
        # Get or create bucket
        if identifier not in self.buckets:
            self.buckets[identifier] = {
                'tokens': self.max_requests,
                'last_refill': current_time
            }
        
        bucket = self.buckets[identifier]
        
        # Refill tokens
        time_passed = current_time - bucket['last_refill']
        tokens_to_add = int(time_passed * (self.max_requests / self.window_seconds))
        
        if tokens_to_add > 0:
            bucket['tokens'] = min(self.max_requests, 
                                 bucket['tokens'] + tokens_to_add)
            bucket['last_refill'] = current_time
        
        # Check and consume token
        if bucket['tokens'] > 0:
            bucket['tokens'] -= 1
            return True
        
        return False
    
    async def _cleanup_expired_buckets(self, current_time: float) -> None:
        """Remove expired rate limit buckets."""
        expired_keys = []
        
        for identifier, bucket in self.buckets.items():
            if current_time - bucket['last_refill'] > self.window_seconds * 2:
                expired_keys.append(identifier)
        
        for key in expired_keys:
            del self.buckets[key]
        
        self.last_cleanup = current_time


class SecurityOrchestrator:
    """Main security orchestrator for the Enhanced CSP Network."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.message_validator = MessageValidator()
        self.tls_config = SecureTLSConfig()
        self.rate_limiter = RateLimiter()
        self.input_validator = SecurityInputValidator()
        
        # Security metrics
        self.security_events = []
        self.blocked_requests = 0
        self.validated_messages = 0
        
        # Configure from config
        self._configure_from_dict(config)
    
    def _configure_from_dict(self, config: Dict[str, Any]) -> None:
        """Configure security settings from dictionary."""
        if 'tls' in config:
            tls_config = config['tls']
            if 'cert_path' in tls_config:
                self.tls_config.cert_path = Path(tls_config['cert_path'])
            if 'key_path' in tls_config:
                self.tls_config.key_path = Path(tls_config['key_path'])
            if 'ca_cert_path' in tls_config:
                self.tls_config.ca_cert_path = Path(tls_config['ca_cert_path'])
        
        if 'rate_limiting' in config:
            rl_config = config['rate_limiting']
            self.rate_limiter = RateLimiter(
                max_requests=rl_config.get('max_requests', MAX_REQUESTS_PER_WINDOW),
                window_seconds=rl_config.get('window_seconds', RATE_LIMIT_WINDOW)
            )
    
    async def validate_incoming_message(self, message: NetworkMessage, 
                                      sender_ip: str) -> bool:
        """Validate incoming message with rate limiting."""
        try:
            # Rate limiting check
            if not await self.rate_limiter.is_allowed(sender_ip):
                self.blocked_requests += 1
                self._log_security_event("rate_limit_exceeded", {"ip": sender_ip})
                return False
            
            # Message validation
            self.message_validator.validate_network_message(message)
            self.validated_messages += 1
            
            return True
            
        except (ValidationError, SecurityError) as e:
            self._log_security_event("validation_failed", {
                "ip": sender_ip,
                "error": str(e),
                "message_type": str(message.type)
            })
            return False
    
    def create_tls_context(self, is_server: bool = False) -> ssl.SSLContext:
        """Create secure TLS context."""
        if is_server:
            return self.tls_config.create_server_context()
        else:
            return self.tls_config.create_client_context()
    
    def _log_security_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log security event for monitoring."""
        event = {
            'timestamp': time.time(),
            'type': event_type,
            'details': details
        }
        self.security_events.append(event)
        
        # Keep only recent events (memory management)
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-500:]
        
        logger.warning(f"Security event: {event_type} - {details}")
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics for monitoring."""
        return {
            'validated_messages': self.validated_messages,
            'blocked_requests': self.blocked_requests,
            'recent_events': len(self.security_events),
            'rate_limiter_buckets': len(self.rate_limiter.buckets)
        }


# Safe import replacement functions
def safe_import_class(module_path: str, class_name: str) -> Any:
    """
    Safe replacement for __import__() with validation.
    
    Args:
        module_path: Dot-separated module path
        class_name: Name of class to import
        
    Returns:
        Imported class
        
    Raises:
        SecurityError: If import is not allowed
        ImportError: If import fails
    """
    # Whitelist of allowed modules
    allowed_modules = {
        'enhanced_csp.network.core.types',
        'enhanced_csp.network.core.config',
        'enhanced_csp.network.p2p.transport',
        'enhanced_csp.network.utils',
    }
    
    if module_path not in allowed_modules:
        raise SecurityError(f"Import not allowed: {module_path}")
    
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to import {class_name} from {module_path}") from e


# Example usage functions
async def example_secure_message_handling():
    """Example of secure message handling."""
    # Initialize security orchestrator
    config = {
        'tls': {
            'cert_path': '/path/to/cert.pem',
            'key_path': '/path/to/key.pem',
            'ca_cert_path': '/path/to/ca.pem'
        },
        'rate_limiting': {
            'max_requests': 100,
            'window_seconds': 60
        }
    }
    
    security = SecurityOrchestrator(config)
    
    # Create mock message
    from ..core.types import NetworkMessage, MessageType, NodeID
    message = NetworkMessage(
        type=MessageType.DATA,
        sender=NodeID("test_node_123"),
        payload=b"Hello, secure world!",
        timestamp=time.time()
    )
    
    # Validate message
    sender_ip = "192.168.1.100"
    is_valid = await security.validate_incoming_message(message, sender_ip)
    
    if is_valid:
        print("Message validated successfully")
        # Process message
    else:
        print("Message validation failed")
    
    # Get security metrics
    metrics = security.get_security_metrics()
    print(f"Security metrics: {metrics}")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_secure_message_handling())
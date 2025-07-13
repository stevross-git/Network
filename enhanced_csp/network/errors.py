# enhanced_csp/network/errors.py
"""Error definitions and handling for Enhanced CSP Network."""

from __future__ import annotations
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict, deque


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of network errors."""
    CONNECTION = "connection"
    PROTOCOL = "protocol"
    SECURITY = "security"
    ROUTING = "routing"
    TRANSPORT = "transport"
    VALIDATION = "validation"
    TIMEOUT = "timeout"
    RESOURCE = "resource"


# Base Network Exception Classes
class NetworkError(Exception):
    """Base class for all network-related errors."""
    
    def __init__(self, message: str, code: Optional[str] = None, 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.code = code or self.__class__.__name__
        self.details = details or {}
        self.timestamp = datetime.utcnow()
        self.severity = ErrorSeverity.MEDIUM
        self.category = ErrorCategory.CONNECTION
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary."""
        return {
            'type': self.__class__.__name__,
            'message': self.message,
            'code': self.code,
            'details': self.details,
            'timestamp': self.timestamp.isoformat(),
            'severity': self.severity.value,
            'category': self.category.value,
        }


class ConnectionError(NetworkError):
    """Raised when connection operations fail."""
    
    def __init__(self, message: str, peer_id: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.peer_id = peer_id
        self.category = ErrorCategory.CONNECTION
        if peer_id:
            self.details['peer_id'] = peer_id


class TimeoutError(NetworkError):
    """Raised when operations timeout."""
    
    def __init__(self, message: str, timeout_duration: Optional[float] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.timeout_duration = timeout_duration
        self.category = ErrorCategory.TIMEOUT
        self.severity = ErrorSeverity.HIGH
        if timeout_duration:
            self.details['timeout_duration'] = timeout_duration


class ProtocolError(NetworkError):
    """Raised when protocol violations occur."""
    
    def __init__(self, message: str, protocol: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.protocol = protocol
        self.category = ErrorCategory.PROTOCOL
        self.severity = ErrorSeverity.HIGH
        if protocol:
            self.details['protocol'] = protocol


class SecurityError(NetworkError):
    """Raised when security violations occur."""
    
    def __init__(self, message: str, threat_type: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.threat_type = threat_type
        self.category = ErrorCategory.SECURITY
        self.severity = ErrorSeverity.CRITICAL
        if threat_type:
            self.details['threat_type'] = threat_type


class ValidationError(NetworkError):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.field = field
        self.value = value
        self.category = ErrorCategory.VALIDATION
        if field:
            self.details['field'] = field
        if value is not None:
            self.details['value'] = str(value)


class RoutingError(NetworkError):
    """Raised when routing operations fail."""
    
    def __init__(self, message: str, destination: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.destination = destination
        self.category = ErrorCategory.ROUTING
        if destination:
            self.details['destination'] = destination


class TransportError(NetworkError):
    """Raised when transport layer operations fail."""
    
    def __init__(self, message: str, transport_type: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.transport_type = transport_type
        self.category = ErrorCategory.TRANSPORT
        if transport_type:
            self.details['transport_type'] = transport_type


class ResourceError(NetworkError):
    """Raised when resource constraints are hit."""
    
    def __init__(self, message: str, resource_type: Optional[str] = None, 
                 current_usage: Optional[float] = None, limit: Optional[float] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.resource_type = resource_type
        self.current_usage = current_usage
        self.limit = limit
        self.category = ErrorCategory.RESOURCE
        self.severity = ErrorSeverity.HIGH
        
        if resource_type:
            self.details['resource_type'] = resource_type
        if current_usage is not None:
            self.details['current_usage'] = current_usage
        if limit is not None:
            self.details['limit'] = limit


# Error Metrics and Tracking
@dataclass
class ErrorMetrics:
    """Tracks error statistics."""
    
    total_errors: int = 0
    errors_by_category: Dict[ErrorCategory, int] = field(default_factory=lambda: defaultdict(int))
    errors_by_severity: Dict[ErrorSeverity, int] = field(default_factory=lambda: defaultdict(int))
    recent_errors: deque = field(default_factory=lambda: deque(maxlen=100))
    error_rate_window: int = 300  # 5 minutes
    
    def record_error(self, error: NetworkError) -> None:
        """Record an error occurrence."""
        self.total_errors += 1
        self.errors_by_category[error.category] += 1
        self.errors_by_severity[error.severity] += 1
        self.recent_errors.append({
            'timestamp': error.timestamp,
            'type': error.__class__.__name__,
            'message': error.message,
            'severity': error.severity.value,
            'category': error.category.value,
        })
        
    def record(self, exc: Exception) -> None:
        """Record an exception occurrence (compatibility method)."""
        # Convert regular Exception to NetworkError for compatibility
        if isinstance(exc, NetworkError):
            self.record_error(exc)
        else:
            # Create a NetworkError wrapper for non-NetworkError exceptions
            wrapper = NetworkError(str(exc))
            wrapper.severity = ErrorSeverity.MEDIUM
            wrapper.category = ErrorCategory.CONNECTION
            self.record_error(wrapper)
    
    def get_error_rate(self) -> float:
        """Calculate recent error rate (errors per minute)."""
        current_time = datetime.utcnow()
        recent_count = 0
        
        for error_record in reversed(self.recent_errors):
            time_diff = (current_time - error_record['timestamp']).total_seconds()
            if time_diff <= self.error_rate_window:
                recent_count += 1
            else:
                break
        
        return (recent_count / self.error_rate_window) * 60  # errors per minute
    
    def get_summary(self) -> Dict[str, Any]:
        """Get error metrics summary."""
        return {
            'total_errors': self.total_errors,
            'error_rate_per_minute': self.get_error_rate(),
            'errors_by_category': {cat.value: count for cat, count in self.errors_by_category.items()},
            'errors_by_severity': {sev.value: count for sev, count in self.errors_by_severity.items()},
            'recent_errors_count': len(self.recent_errors),
        }


# Circuit Breaker Pattern for Error Handling
class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpen(NetworkError):
    """Raised when circuit breaker is open."""
    
    def __init__(self, service: str, **kwargs):
        super().__init__(f"Circuit breaker open for service: {service}", **kwargs)
        self.service = service
        self.severity = ErrorSeverity.HIGH
        self.details['service'] = service


@dataclass
class CircuitBreaker:
    """Circuit breaker for handling cascading failures."""
    
    name: str
    failure_threshold: int = 5
    timeout: float = 60.0  # seconds
    success_threshold: int = 3
    
    # Internal state
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
            else:
                raise CircuitBreakerOpen(self.name)
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.timeout
    
    def _on_success(self) -> None:
        """Handle successful operation."""
        self.failure_count = 0
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.success_count = 0
    
    def _on_failure(self) -> None:
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.success_count = 0
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN


# Error Handler Registry
class ErrorHandler:
    """Base class for error handlers."""
    
    def can_handle(self, error: NetworkError) -> bool:
        """Check if this handler can handle the error."""
        return True
    
    def handle(self, error: NetworkError) -> None:
        """Handle the error."""
        pass


class LoggingErrorHandler(ErrorHandler):
    """Error handler that logs errors."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def handle(self, error: NetworkError) -> None:
        """Log the error with appropriate level."""
        level_map = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL,
        }
        
        level = level_map.get(error.severity, logging.ERROR)
        self.logger.log(level, f"{error.category.value.upper()}: {error.message}", 
                       extra={'error_details': error.details})


class MetricsErrorHandler(ErrorHandler):
    """Error handler that records metrics."""
    
    def __init__(self, metrics: ErrorMetrics):
        self.metrics = metrics
    
    def handle(self, error: NetworkError) -> None:
        """Record error in metrics."""
        self.metrics.record_error(error)


class ErrorHandlerRegistry:
    """Registry for error handlers."""
    
    def __init__(self):
        self.handlers: List[ErrorHandler] = []
        self.metrics = ErrorMetrics()
        
        # Add default handlers
        self.add_handler(LoggingErrorHandler())
        self.add_handler(MetricsErrorHandler(self.metrics))
    
    def add_handler(self, handler: ErrorHandler) -> None:
        """Add an error handler."""
        self.handlers.append(handler)
    
    def handle_error(self, error: NetworkError) -> None:
        """Handle an error with all registered handlers."""
        for handler in self.handlers:
            if handler.can_handle(error):
                try:
                    handler.handle(error)
                except Exception as e:
                    # Don't let error handling fail
                    logging.error(f"Error handler failed: {e}")
    
    def get_metrics(self) -> ErrorMetrics:
        """Get error metrics."""
        return self.metrics


# Global error handler registry
_global_error_registry = ErrorHandlerRegistry()


def handle_error(error: NetworkError) -> None:
    """Handle an error using the global registry."""
    _global_error_registry.handle_error(error)


def get_error_metrics() -> ErrorMetrics:
    """Get global error metrics."""
    return _global_error_registry.get_metrics()


def add_error_handler(handler: ErrorHandler) -> None:
    """Add an error handler to the global registry."""
    _global_error_registry.add_handler(handler)


# Convenience functions for creating common errors
def connection_failed(peer_id: str, reason: str) -> ConnectionError:
    """Create a connection failed error."""
    return ConnectionError(f"Connection to {peer_id} failed: {reason}", peer_id=peer_id)


def operation_timeout(operation: str, timeout: float) -> TimeoutError:
    """Create an operation timeout error."""
    return TimeoutError(f"Operation '{operation}' timed out after {timeout}s", 
                       timeout_duration=timeout)


def invalid_message(reason: str, field: Optional[str] = None) -> ValidationError:
    """Create an invalid message error."""
    return ValidationError(f"Invalid message: {reason}", field=field)


def route_not_found(destination: str) -> RoutingError:
    """Create a route not found error."""
    return RoutingError(f"No route to destination: {destination}", destination=destination)


def security_violation(threat: str, details: Optional[str] = None) -> SecurityError:
    """Create a security violation error."""
    message = f"Security violation: {threat}"
    if details:
        message += f" - {details}"
    return SecurityError(message, threat_type=threat)


# Export all error types and utilities
__all__ = [
    # Base error classes
    "NetworkError",
    "ConnectionError", 
    "TimeoutError",
    "ProtocolError",
    "SecurityError",
    "ValidationError",
    "RoutingError",
    "TransportError",
    "ResourceError",
    
    # Enums
    "ErrorSeverity",
    "ErrorCategory",
    
    # Metrics and tracking
    "ErrorMetrics",
    
    # Circuit breaker
    "CircuitBreaker",
    "CircuitBreakerOpen",
    "CircuitBreakerState",
    
    # Error handling
    "ErrorHandler",
    "LoggingErrorHandler",
    "MetricsErrorHandler",
    "ErrorHandlerRegistry",
    
    # Global functions
    "handle_error",
    "get_error_metrics",
    "add_error_handler",
    
    # Convenience functions
    "connection_failed",
    "operation_timeout",
    "invalid_message",
    "route_not_found",
    "security_violation",
]
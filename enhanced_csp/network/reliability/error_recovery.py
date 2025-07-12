# enhanced_csp/network/reliability/error_recovery.py
"""
Comprehensive error recovery and reliability system for Enhanced CSP Network.
Implements retry logic, failure detection, and self-healing capabilities.
"""

import asyncio
import logging
import time
import random
import traceback
from typing import Dict, List, Optional, Any, Callable, TypeVar, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import functools
import threading

from ..errors import NetworkError, ConnectionError, TimeoutError, SecurityError, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar('T')

class FailureType(Enum):
    """Types of failures that can occur."""
    CONNECTION_FAILED = "connection_failed"
    TIMEOUT = "timeout"
    AUTHENTICATION_FAILED = "authentication_failed"
    PROTOCOL_ERROR = "protocol_error"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    SERVICE_UNAVAILABLE = "service_unavailable"
    UNKNOWN = "unknown"


class RecoveryAction(Enum):
    """Types of recovery actions."""
    RETRY = "retry"
    RECONNECT = "reconnect"
    FALLBACK = "fallback"
    ESCALATE = "escalate"
    IGNORE = "ignore"
    CIRCUIT_BREAK = "circuit_break"


@dataclass
class FailureEvent:
    """Represents a failure event."""
    timestamp: float
    failure_type: FailureType
    service: str
    error_message: str
    context: Dict[str, Any]
    retry_count: int = 0
    resolved: bool = False


@dataclass
class RecoveryStrategy:
    """Defines how to recover from specific failure types."""
    failure_type: FailureType
    max_retries: int
    base_delay: float
    max_delay: float
    backoff_multiplier: float
    jitter: bool
    recovery_actions: List[RecoveryAction]
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        delay = min(self.base_delay * (self.backoff_multiplier ** attempt), self.max_delay)
        
        if self.jitter:
            # Add random jitter to prevent thundering herd
            jitter_amount = delay * 0.1
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        return max(0, delay)


class RetryManager:
    """Manages retry logic with exponential backoff and jitter."""
    
    def __init__(self):
        self.strategies: Dict[FailureType, RecoveryStrategy] = {
            FailureType.CONNECTION_FAILED: RecoveryStrategy(
                failure_type=FailureType.CONNECTION_FAILED,
                max_retries=5,
                base_delay=1.0,
                max_delay=30.0,
                backoff_multiplier=2.0,
                jitter=True,
                recovery_actions=[RecoveryAction.RETRY, RecoveryAction.RECONNECT]
            ),
            FailureType.TIMEOUT: RecoveryStrategy(
                failure_type=FailureType.TIMEOUT,
                max_retries=3,
                base_delay=2.0,
                max_delay=20.0,
                backoff_multiplier=1.5,
                jitter=True,
                recovery_actions=[RecoveryAction.RETRY, RecoveryAction.FALLBACK]
            ),
            FailureType.AUTHENTICATION_FAILED: RecoveryStrategy(
                failure_type=FailureType.AUTHENTICATION_FAILED,
                max_retries=2,
                base_delay=5.0,
                max_delay=60.0,
                backoff_multiplier=3.0,
                jitter=False,
                recovery_actions=[RecoveryAction.ESCALATE]
            ),
            FailureType.RESOURCE_EXHAUSTED: RecoveryStrategy(
                failure_type=FailureType.RESOURCE_EXHAUSTED,
                max_retries=10,
                base_delay=0.5,
                max_delay=5.0,
                backoff_multiplier=1.2,
                jitter=True,
                recovery_actions=[RecoveryAction.RETRY, RecoveryAction.CIRCUIT_BREAK]
            ),
            FailureType.SERVICE_UNAVAILABLE: RecoveryStrategy(
                failure_type=FailureType.SERVICE_UNAVAILABLE,
                max_retries=8,
                base_delay=3.0,
                max_delay=45.0,
                backoff_multiplier=2.0,
                jitter=True,
                recovery_actions=[RecoveryAction.RETRY, RecoveryAction.FALLBACK]
            ),
        }
        
        self.retry_stats = defaultdict(lambda: {'attempts': 0, 'successes': 0, 'failures': 0})
        self._lock = threading.Lock()
    
    async def execute_with_retry(self, func: Callable[..., T], *args, 
                               service: str = "unknown", **kwargs) -> T:
        """Execute function with automatic retry logic."""
        last_exception = None
        
        for attempt in range(self._get_max_retries_for_func(func)):
            try:
                with self._lock:
                    self.retry_stats[service]['attempts'] += 1
                
                # Execute the function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                with self._lock:
                    self.retry_stats[service]['successes'] += 1
                
                if attempt > 0:
                    logger.info(f"Operation succeeded after {attempt + 1} attempts: {service}")
                
                return result
                
            except Exception as e:
                last_exception = e
                failure_type = self._classify_failure(e)
                strategy = self.strategies.get(failure_type, self._get_default_strategy())
                
                with self._lock:
                    self.retry_stats[service]['failures'] += 1
                
                if attempt >= strategy.max_retries:
                    logger.error(f"Max retries exceeded for {service}: {e}")
                    break
                
                delay = strategy.calculate_delay(attempt)
                logger.warning(f"Attempt {attempt + 1} failed for {service}: {e}. "
                             f"Retrying in {delay:.2f}s")
                
                await asyncio.sleep(delay)
        
        # All retries exhausted
        raise last_exception or Exception("Retry limit exceeded")
    
    def _classify_failure(self, exception: Exception) -> FailureType:
        """Classify exception to determine failure type."""
        if isinstance(exception, ConnectionError):
            return FailureType.CONNECTION_FAILED
        elif isinstance(exception, TimeoutError):
            return FailureType.TIMEOUT
        elif isinstance(exception, SecurityError):
            return FailureType.AUTHENTICATION_FAILED
        elif isinstance(exception, ValidationError):
            return FailureType.PROTOCOL_ERROR
        elif "resource" in str(exception).lower() or "limit" in str(exception).lower():
            return FailureType.RESOURCE_EXHAUSTED
        elif "unavailable" in str(exception).lower() or "not found" in str(exception).lower():
            return FailureType.SERVICE_UNAVAILABLE
        else:
            return FailureType.UNKNOWN
    
    def _get_max_retries_for_func(self, func: Callable) -> int:
        """Get max retries based on function or default."""
        # Check if function has retry annotation
        if hasattr(func, '__retry_config__'):
            return func.__retry_config__.get('max_retries', 3)
        return 3
    
    def _get_default_strategy(self) -> RecoveryStrategy:
        """Get default recovery strategy."""
        return RecoveryStrategy(
            failure_type=FailureType.UNKNOWN,
            max_retries=3,
            base_delay=1.0,
            max_delay=10.0,
            backoff_multiplier=2.0,
            jitter=True,
            recovery_actions=[RecoveryAction.RETRY]
        )
    
    def get_retry_stats(self) -> Dict[str, Dict[str, int]]:
        """Get retry statistics."""
        with self._lock:
            return dict(self.retry_stats)


class HealthChecker:
    """Health checking system for services and endpoints."""
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.health_status: Dict[str, bool] = {}
        self.health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.health_checkers: Dict[str, Callable] = {}
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        self._lock = threading.RLock()
    
    def register_health_check(self, service: str, health_check_func: Callable[[], bool]) -> None:
        """Register a health check function for a service."""
        with self._lock:
            self.health_checkers[service] = health_check_func
            
            # Start monitoring task if not already running
            if service not in self.monitoring_tasks:
                task = asyncio.create_task(self._monitor_service_health(service))
                self.monitoring_tasks[service] = task
    
    async def _monitor_service_health(self, service: str) -> None:
        """Monitor health of a specific service."""
        while service in self.health_checkers:
            try:
                health_func = self.health_checkers[service]
                
                # Execute health check
                if asyncio.iscoroutinefunction(health_func):
                    is_healthy = await health_func()
                else:
                    is_healthy = health_func()
                
                with self._lock:
                    old_status = self.health_status.get(service, True)
                    self.health_status[service] = is_healthy
                    self.health_history[service].append((time.time(), is_healthy))
                
                # Log status changes
                if old_status != is_healthy:
                    if is_healthy:
                        logger.info(f"Service {service} is now healthy")
                    else:
                        logger.warning(f"Service {service} is now unhealthy")
                
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check failed for {service}: {e}")
                with self._lock:
                    self.health_status[service] = False
                    self.health_history[service].append((time.time(), False))
                
                await asyncio.sleep(self.check_interval)
    
    def is_healthy(self, service: str) -> bool:
        """Check if service is currently healthy."""
        with self._lock:
            return self.health_status.get(service, False)
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary for all services."""
        with self._lock:
            summary = {}
            
            for service in self.health_checkers:
                history = list(self.health_history[service])
                
                if history:
                    recent_checks = [h[1] for h in history[-10:]]  # Last 10 checks
                    uptime_ratio = sum(recent_checks) / len(recent_checks)
                    last_check_time = history[-1][0]
                    last_status = history[-1][1]
                else:
                    uptime_ratio = 0.0
                    last_check_time = None
                    last_status = False
                
                summary[service] = {
                    'current_status': self.health_status.get(service, False),
                    'uptime_ratio': uptime_ratio,
                    'last_check': last_check_time,
                    'last_status': last_status,
                    'total_checks': len(history)
                }
            
            return summary
    
    async def shutdown(self) -> None:
        """Shutdown health monitoring."""
        with self._lock:
            tasks = list(self.monitoring_tasks.values())
            self.monitoring_tasks.clear()
        
        for task in tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


class FallbackManager:
    """Manages fallback mechanisms when primary services fail."""
    
    def __init__(self):
        self.fallback_chains: Dict[str, List[Callable]] = {}
        self.fallback_stats = defaultdict(lambda: {'primary_calls': 0, 'fallback_calls': 0})
        self._lock = threading.Lock()
    
    def register_fallback_chain(self, service: str, primary: Callable, 
                              fallbacks: List[Callable]) -> None:
        """Register a fallback chain for a service."""
        with self._lock:
            self.fallback_chains[service] = [primary] + fallbacks
    
    async def execute_with_fallback(self, service: str, *args, **kwargs) -> Any:
        """Execute service with fallback chain."""
        if service not in self.fallback_chains:
            raise ValueError(f"No fallback chain registered for service: {service}")
        
        chain = self.fallback_chains[service]
        last_exception = None
        
        for i, func in enumerate(chain):
            try:
                with self._lock:
                    if i == 0:
                        self.fallback_stats[service]['primary_calls'] += 1
                    else:
                        self.fallback_stats[service]['fallback_calls'] += 1
                
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                if i > 0:
                    logger.info(f"Fallback {i} succeeded for service {service}")
                
                return result
                
            except Exception as e:
                last_exception = e
                if i < len(chain) - 1:
                    logger.warning(f"Service {service} attempt {i + 1} failed: {e}. "
                                 f"Trying fallback...")
                else:
                    logger.error(f"All fallbacks exhausted for service {service}: {e}")
        
        raise last_exception or Exception("All fallbacks failed")
    
    def get_fallback_stats(self) -> Dict[str, Dict[str, int]]:
        """Get fallback usage statistics."""
        with self._lock:
            return dict(self.fallback_stats)


class ErrorRecoveryManager:
    """Main error recovery manager coordinating all recovery mechanisms."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.retry_manager = RetryManager()
        self.health_checker = HealthChecker(
            check_interval=config.get('health_check_interval', 30.0)
        )
        self.fallback_manager = FallbackManager()
        
        # Failure tracking
        self.failure_events: deque = deque(maxlen=1000)
        self.recovery_strategies = {
            ConnectionError: self._handle_connection_error,
            TimeoutError: self._handle_timeout_error,
            SecurityError: self._handle_security_error,
            ValidationError: self._handle_validation_error,
        }
        
        # Self-healing mechanisms
        self.auto_healing_enabled = config.get('auto_healing', True)
        self.healing_tasks: List[asyncio.Task] = []
        
        # Metrics
        self.recovery_metrics = {
            'total_failures': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'auto_heals': 0
        }
        
        self._lock = threading.Lock()
    
    async def handle_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """
        Handle error with appropriate recovery strategy.
        
        Returns:
            True if error was handled/recovered, False otherwise
        """
        with self._lock:
            self.recovery_metrics['total_failures'] += 1
        
        # Create failure event
        failure_event = FailureEvent(
            timestamp=time.time(),
            failure_type=self.retry_manager._classify_failure(error),
            service=context.get('service', 'unknown'),
            error_message=str(error),
            context=context.copy()
        )
        
        self.failure_events.append(failure_event)
        
        # Try to recover
        recovery_func = self.recovery_strategies.get(type(error), self._handle_generic_error)
        
        try:
            recovered = await recovery_func(error, context)
            
            with self._lock:
                if recovered:
                    self.recovery_metrics['successful_recoveries'] += 1
                    failure_event.resolved = True
                else:
                    self.recovery_metrics['failed_recoveries'] += 1
            
            return recovered
            
        except Exception as recovery_error:
            logger.error(f"Recovery strategy failed: {recovery_error}")
            with self._lock:
                self.recovery_metrics['failed_recoveries'] += 1
            return False
    
    async def _handle_connection_error(self, error: ConnectionError, 
                                     context: Dict[str, Any]) -> bool:
        """Handle connection errors with reconnection logic."""
        service = context.get('service', 'unknown')
        endpoint = context.get('endpoint')
        
        logger.warning(f"Handling connection error for {service}: {error}")
        
        # Check if we have fallback endpoints
        fallback_endpoints = context.get('fallback_endpoints', [])
        
        if fallback_endpoints:
            for fallback_endpoint in fallback_endpoints:
                try:
                    # Try to establish connection to fallback
                    connect_func = context.get('connect_function')
                    if connect_func:
                        if asyncio.iscoroutinefunction(connect_func):
                            await connect_func(fallback_endpoint)
                        else:
                            connect_func(fallback_endpoint)
                        
                        logger.info(f"Successfully connected to fallback endpoint: {fallback_endpoint}")
                        return True
                        
                except Exception as fallback_error:
                    logger.warning(f"Fallback endpoint {fallback_endpoint} also failed: {fallback_error}")
                    continue
        
        # Try reconnection with exponential backoff
        reconnect_func = context.get('reconnect_function')
        if reconnect_func:
            try:
                if asyncio.iscoroutinefunction(reconnect_func):
                    await reconnect_func()
                else:
                    reconnect_func()
                
                logger.info(f"Successfully reconnected to {service}")
                return True
                
            except Exception as reconnect_error:
                logger.error(f"Reconnection failed for {service}: {reconnect_error}")
        
        return False
    
    async def _handle_timeout_error(self, error: TimeoutError, 
                                  context: Dict[str, Any]) -> bool:
        """Handle timeout errors with adaptive timeout adjustment."""
        service = context.get('service', 'unknown')
        
        # Increase timeout for future operations
        current_timeout = context.get('timeout', 30.0)
        new_timeout = min(current_timeout * 1.5, 300.0)  # Cap at 5 minutes
        
        # Store adjusted timeout
        if 'timeout_adjustment_func' in context:
            adjust_func = context['timeout_adjustment_func']
            try:
                if asyncio.iscoroutinefunction(adjust_func):
                    await adjust_func(new_timeout)
                else:
                    adjust_func(new_timeout)
                
                logger.info(f"Adjusted timeout for {service} from {current_timeout}s to {new_timeout}s")
                return True
                
            except Exception as adjust_error:
                logger.error(f"Failed to adjust timeout for {service}: {adjust_error}")
        
        return False
    
    async def _handle_security_error(self, error: SecurityError, 
                                   context: Dict[str, Any]) -> bool:
        """Handle security errors with authentication refresh."""
        service = context.get('service', 'unknown')
        
        # Try to refresh authentication
        refresh_func = context.get('auth_refresh_function')
        if refresh_func:
            try:
                if asyncio.iscoroutinefunction(refresh_func):
                    await refresh_func()
                else:
                    refresh_func()
                
                logger.info(f"Successfully refreshed authentication for {service}")
                return True
                
            except Exception as refresh_error:
                logger.error(f"Authentication refresh failed for {service}: {refresh_error}")
        
        # If refresh fails, escalate to security team
        await self._escalate_security_incident(error, context)
        return False
    
    async def _handle_validation_error(self, error: ValidationError, 
                                     context: Dict[str, Any]) -> bool:
        """Handle validation errors with data sanitization."""
        service = context.get('service', 'unknown')
        
        # Try to sanitize and retry with clean data
        sanitize_func = context.get('data_sanitize_function')
        if sanitize_func:
            try:
                if asyncio.iscoroutinefunction(sanitize_func):
                    sanitized_data = await sanitize_func()
                else:
                    sanitized_data = sanitize_func()
                
                # Retry operation with sanitized data
                retry_func = context.get('retry_function')
                if retry_func:
                    if asyncio.iscoroutinefunction(retry_func):
                        await retry_func(sanitized_data)
                    else:
                        retry_func(sanitized_data)
                    
                    logger.info(f"Successfully retried {service} with sanitized data")
                    return True
                
            except Exception as sanitize_error:
                logger.error(f"Data sanitization failed for {service}: {sanitize_error}")
        
        return False
    
    async def _handle_generic_error(self, error: Exception, 
                                  context: Dict[str, Any]) -> bool:
        """Handle generic errors with basic retry logic."""
        service = context.get('service', 'unknown')
        
        # Basic retry with exponential backoff
        max_retries = context.get('max_retries', 3)
        current_retry = context.get('current_retry', 0)
        
        if current_retry < max_retries:
            delay = min(2 ** current_retry, 30)  # Cap at 30 seconds
            await asyncio.sleep(delay)
            
            context['current_retry'] = current_retry + 1
            logger.info(f"Retrying {service} (attempt {current_retry + 1}/{max_retries})")
            return True
        
        return False
    
    async def _escalate_security_incident(self, error: SecurityError, 
                                        context: Dict[str, Any]) -> None:
        """Escalate security incident to monitoring systems."""
        incident = {
            'timestamp': time.time(),
            'type': 'security_error',
            'error': str(error),
            'context': context,
            'stack_trace': traceback.format_exc()
        }
        
        # Log security incident
        logger.critical(f"Security incident escalated: {incident}")
        
        # In production, this would integrate with SIEM/monitoring systems
        # Example: send to security team, trigger alerts, etc.
    
    async def start_auto_healing(self) -> None:
        """Start automatic healing processes."""
        if not self.auto_healing_enabled:
            return
        
        # Start healing tasks
        self.healing_tasks = [
            asyncio.create_task(self._auto_heal_connections()),
            asyncio.create_task(self._auto_heal_performance()),
            asyncio.create_task(self._auto_heal_resources())
        ]
        
        logger.info("Auto-healing processes started")
    
    async def stop_auto_healing(self) -> None:
        """Stop automatic healing processes."""
        for task in self.healing_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self.healing_tasks.clear()
        logger.info("Auto-healing processes stopped")
    
    async def _auto_heal_connections(self) -> None:
        """Automatically heal connection issues."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Analyze recent connection failures
                recent_failures = [
                    event for event in self.failure_events
                    if (time.time() - event.timestamp < 300 and  # Last 5 minutes
                        event.failure_type == FailureType.CONNECTION_FAILED and
                        not event.resolved)
                ]
                
                if len(recent_failures) >= 3:
                    logger.info("Detected connection issues, initiating auto-healing")
                    with self._lock:
                        self.recovery_metrics['auto_heals'] += 1
                    
                    # Trigger connection pool refresh
                    # This would integrate with your connection pool
                    await self._refresh_connection_pool()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-heal connections error: {e}")
    
    async def _auto_heal_performance(self) -> None:
        """Automatically heal performance issues."""
        while True:
            try:
                await asyncio.sleep(120)  # Check every 2 minutes
                
                # Analyze performance metrics
                # This would integrate with your performance monitoring
                await self._optimize_performance_if_needed()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-heal performance error: {e}")
    
    async def _auto_heal_resources(self) -> None:
        """Automatically heal resource exhaustion issues."""
        while True:
            try:
                await asyncio.sleep(180)  # Check every 3 minutes
                
                # Check for resource exhaustion patterns
                resource_failures = [
                    event for event in self.failure_events
                    if (time.time() - event.timestamp < 600 and  # Last 10 minutes
                        event.failure_type == FailureType.RESOURCE_EXHAUSTED)
                ]
                
                if len(resource_failures) >= 5:
                    logger.info("Detected resource exhaustion, initiating cleanup")
                    await self._cleanup_resources()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-heal resources error: {e}")
    
    async def _refresh_connection_pool(self) -> None:
        """Refresh connection pool to resolve connection issues."""
        # This would integrate with your actual connection pool
        logger.info("Refreshing connection pool for auto-healing")
    
    async def _optimize_performance_if_needed(self) -> None:
        """Optimize performance if degradation is detected."""
        # This would integrate with your performance optimization systems
        logger.debug("Checking performance optimization opportunities")
    
    async def _cleanup_resources(self) -> None:
        """Clean up resources to resolve exhaustion issues."""
        # Trigger garbage collection and resource cleanup
        import gc
        collected = gc.collect()
        logger.info(f"Auto-healing: Garbage collected {collected} objects")
    
    def get_recovery_metrics(self) -> Dict[str, Any]:
        """Get comprehensive recovery metrics."""
        with self._lock:
            recent_failures = [
                event for event in self.failure_events
                if time.time() - event.timestamp < 3600  # Last hour
            ]
            
            failure_by_type = defaultdict(int)
            for event in recent_failures:
                failure_by_type[event.failure_type.value] += 1
            
            return {
                'total_metrics': self.recovery_metrics.copy(),
                'recent_failures': len(recent_failures),
                'failure_by_type': dict(failure_by_type),
                'retry_stats': self.retry_manager.get_retry_stats(),
                'fallback_stats': self.fallback_manager.get_fallback_stats(),
                'health_summary': self.health_checker.get_health_summary(),
                'auto_healing_active': len(self.healing_tasks) > 0
            }


# Decorator for automatic retry
def with_retry(max_retries: int = 3, service: str = None):
    """Decorator to add automatic retry logic to functions."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            retry_manager = RetryManager()
            return await retry_manager.execute_with_retry(
                func, *args, service=service or func.__name__, **kwargs
            )
        
        # Add retry configuration to function
        wrapper.__retry_config__ = {'max_retries': max_retries}
        return wrapper
    
    return decorator


# Example usage
@with_retry(max_retries=5, service="network_operation")
async def example_network_operation(endpoint: str) -> str:
    """Example network operation with automatic retry."""
    # Simulate network operation that might fail
    if random.random() < 0.3:  # 30% failure rate
        raise ConnectionError("Network operation failed")
    
    return f"Connected to {endpoint}"


async def example_error_recovery():
    """Example of comprehensive error recovery usage."""
    config = {
        'health_check_interval': 30.0,
        'auto_healing': True
    }
    
    recovery_manager = ErrorRecoveryManager(config)
    
    # Register health checks
    async def check_database_health():
        # Simulate database health check
        return random.random() > 0.1  # 90% healthy
    
    recovery_manager.health_checker.register_health_check(
        "database", check_database_health
    )
    
    # Register fallback chain
    async def primary_service():
        raise ConnectionError("Primary service failed")
    
    async def fallback_service():
        return "Fallback service response"
    
    recovery_manager.fallback_manager.register_fallback_chain(
        "example_service", primary_service, [fallback_service]
    )
    
    # Start auto-healing
    await recovery_manager.start_auto_healing()
    
    try:
        # Test error handling
        context = {
            'service': 'test_service',
            'endpoint': 'test_endpoint',
            'timeout': 30.0
        }
        
        error = ConnectionError("Test connection error")
        recovered = await recovery_manager.handle_error(error, context)
        print(f"Error recovery result: {recovered}")
        
        # Test fallback execution
        result = await recovery_manager.fallback_manager.execute_with_fallback(
            "example_service"
        )
        print(f"Fallback result: {result}")
        
        # Test retry decorator
        result = await example_network_operation("test.example.com")
        print(f"Network operation result: {result}")
        
        # Get metrics
        metrics = recovery_manager.get_recovery_metrics()
        print(f"Recovery metrics: {metrics}")
        
    finally:
        await recovery_manager.stop_auto_healing()
        await recovery_manager.health_checker.shutdown()


if __name__ == "__main__":
    asyncio.run(example_error_recovery())
# enhanced_csp/network/performance/optimization_engine.py
"""
Advanced performance optimization engine for Enhanced CSP Network.
Implements adaptive connection pooling, memory management, and circuit breaker patterns.
"""

import asyncio
import time
import weakref
import threading
import statistics
from typing import Dict, List, Optional, Any, Callable, TypeVar, Generic, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import logging
import gc
import psutil
import os

logger = logging.getLogger(__name__)

T = TypeVar('T')

class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class PerformanceMetrics:
    """Performance tracking metrics."""
    start_time: float = field(default_factory=time.time)
    operations_count: int = 0
    error_count: int = 0
    total_latency: float = 0.0
    max_latency: float = 0.0
    min_latency: float = float('inf')
    
    def add_operation(self, latency: float, error: bool = False) -> None:
        """Record an operation."""
        self.operations_count += 1
        if error:
            self.error_count += 1
        
        self.total_latency += latency
        self.max_latency = max(self.max_latency, latency)
        self.min_latency = min(self.min_latency, latency)
    
    @property
    def average_latency(self) -> float:
        """Calculate average latency."""
        return self.total_latency / self.operations_count if self.operations_count > 0 else 0.0
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        return self.error_count / self.operations_count if self.operations_count > 0 else 0.0
    
    @property
    def operations_per_second(self) -> float:
        """Calculate operations per second."""
        elapsed = time.time() - self.start_time
        return self.operations_count / elapsed if elapsed > 0 else 0.0


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60, 
                 success_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.success_threshold = success_threshold
        
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitBreakerState.CLOSED
        self._lock = threading.Lock()
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if (self.last_failure_time and 
                    time.time() - self.last_failure_time > self.timeout):
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.success_count = 0
                else:
                    raise CircuitBreakerOpen("Circuit breaker is open")
        
        try:
            start_time = time.perf_counter()
            
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            latency = time.perf_counter() - start_time
            
            with self._lock:
                if self.state == CircuitBreakerState.HALF_OPEN:
                    self.success_count += 1
                    if self.success_count >= self.success_threshold:
                        self.state = CircuitBreakerState.CLOSED
                        self.failure_count = 0
                elif self.state == CircuitBreakerState.CLOSED:
                    self.failure_count = max(0, self.failure_count - 1)
            
            return result
            
        except Exception as e:
            with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitBreakerState.OPEN
                    logger.warning(f"Circuit breaker opened due to {self.failure_count} failures")
            
            raise e


class CircuitBreakerOpen(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class BoundedCache(Generic[T]):
    """Memory-efficient bounded cache with TTL and LRU eviction."""
    
    def __init__(self, max_size: int, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[str, T] = {}
        self.access_times: Dict[str, float] = {}
        self.insertion_order: deque = deque()
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[T]:
        """Get item from cache."""
        with self._lock:
            if key not in self.cache:
                return None
            
            # Check TTL
            if time.time() - self.access_times[key] > self.ttl:
                self._remove_key(key)
                return None
            
            # Update access time
            self.access_times[key] = time.time()
            
            # Move to end for LRU
            if key in self.insertion_order:
                self.insertion_order.remove(key)
            self.insertion_order.append(key)
            
            return self.cache[key]
    
    def put(self, key: str, value: T) -> None:
        """Put item in cache."""
        with self._lock:
            current_time = time.time()
            
            # Update existing key
            if key in self.cache:
                self.cache[key] = value
                self.access_times[key] = current_time
                if key in self.insertion_order:
                    self.insertion_order.remove(key)
                self.insertion_order.append(key)
                return
            
            # Evict if necessary
            while len(self.cache) >= self.max_size:
                self._evict_lru()
            
            # Add new item
            self.cache[key] = value
            self.access_times[key] = current_time
            self.insertion_order.append(key)
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self.insertion_order:
            return
        
        key = self.insertion_order.popleft()
        self._remove_key(key)
    
    def _remove_key(self, key: str) -> None:
        """Remove key from cache."""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        if key in self.insertion_order:
            self.insertion_order.remove(key)
    
    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns number of removed entries."""
        with self._lock:
            current_time = time.time()
            expired_keys = []
            
            for key, access_time in self.access_times.items():
                if current_time - access_time > self.ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_key(key)
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'utilization': len(self.cache) / self.max_size,
                'oldest_access': min(self.access_times.values()) if self.access_times else None
            }


class AdaptiveConnectionPool:
    """Adaptive connection pool with dynamic sizing and load balancing."""
    
    def __init__(self, base_size: int = 10, max_size: int = 100, 
                 min_size: int = 5):
        self.base_size = base_size
        self.max_size = max_size
        self.min_size = min_size
        self.current_size = base_size
        
        self.connections: Dict[str, List[Any]] = defaultdict(list)
        self.connection_stats: Dict[str, PerformanceMetrics] = {}
        self.load_history: deque = deque(maxlen=100)
        self.pool_stats = {
            'total_connections': 0,
            'active_connections': 0,
            'pool_hits': 0,
            'pool_misses': 0,
            'expansions': 0,
            'contractions': 0
        }
        
        self._lock = threading.RLock()
        self._adjustment_task: Optional[asyncio.Task] = None
        
    async def start_adaptive_sizing(self) -> None:
        """Start adaptive pool sizing task."""
        if self._adjustment_task is None:
            self._adjustment_task = asyncio.create_task(self._adaptive_sizing_loop())
    
    async def stop_adaptive_sizing(self) -> None:
        """Stop adaptive pool sizing task."""
        if self._adjustment_task:
            self._adjustment_task.cancel()
            try:
                await self._adjustment_task
            except asyncio.CancelledError:
                pass
            self._adjustment_task = None
    
    async def get_connection(self, endpoint: str, factory: Callable) -> Any:
        """Get connection from pool or create new one."""
        with self._lock:
            # Try to get existing connection
            if endpoint in self.connections and self.connections[endpoint]:
                connection = self.connections[endpoint].pop()
                self.pool_stats['pool_hits'] += 1
                self.pool_stats['active_connections'] += 1
                return connection
            
            # Create new connection
            self.pool_stats['pool_misses'] += 1
        
        try:
            if asyncio.iscoroutinefunction(factory):
                connection = await factory(endpoint)
            else:
                connection = factory(endpoint)
            
            with self._lock:
                self.pool_stats['total_connections'] += 1
                self.pool_stats['active_connections'] += 1
            
            return connection
            
        except Exception as e:
            logger.error(f"Failed to create connection to {endpoint}: {e}")
            raise
    
    async def return_connection(self, endpoint: str, connection: Any) -> None:
        """Return connection to pool."""
        with self._lock:
            if len(self.connections[endpoint]) < self.current_size:
                self.connections[endpoint].append(connection)
            else:
                # Pool is full, close connection
                if hasattr(connection, 'close'):
                    if asyncio.iscoroutinefunction(connection.close):
                        await connection.close()
                    else:
                        connection.close()
            
            self.pool_stats['active_connections'] = max(0, 
                self.pool_stats['active_connections'] - 1)
    
    async def _adaptive_sizing_loop(self) -> None:
        """Background task for adaptive pool sizing."""
        while True:
            try:
                await asyncio.sleep(30)  # Adjust every 30 seconds
                await self._adjust_pool_size()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in adaptive sizing: {e}")
    
    async def _adjust_pool_size(self) -> None:
        """Adjust pool size based on load metrics."""
        current_load = self._calculate_current_load()
        self.load_history.append(current_load)
        
        if len(self.load_history) < 10:
            return  # Need more data
        
        avg_load = statistics.mean(self.load_history)
        load_variance = statistics.variance(self.load_history)
        
        with self._lock:
            old_size = self.current_size
            
            # High load and low variance -> increase pool
            if avg_load > 0.8 and load_variance < 0.1:
                if self.current_size < self.max_size:
                    self.current_size = min(self.max_size, 
                                          int(self.current_size * 1.2))
                    self.pool_stats['expansions'] += 1
            
            # Low load -> decrease pool
            elif avg_load < 0.3:
                if self.current_size > self.min_size:
                    self.current_size = max(self.min_size, 
                                          int(self.current_size * 0.8))
                    self.pool_stats['contractions'] += 1
                    
                    # Remove excess connections
                    await self._remove_excess_connections()
            
            if old_size != self.current_size:
                logger.info(f"Adjusted pool size from {old_size} to {self.current_size} "
                          f"(load: {avg_load:.2f}, variance: {load_variance:.2f})")
    
    def _calculate_current_load(self) -> float:
        """Calculate current pool load (0.0 to 1.0)."""
        with self._lock:
            if self.pool_stats['total_connections'] == 0:
                return 0.0
            
            return (self.pool_stats['active_connections'] / 
                   max(1, self.pool_stats['total_connections']))
    
    async def _remove_excess_connections(self) -> None:
        """Remove excess connections from pool."""
        with self._lock:
            for endpoint, connections in self.connections.items():
                while len(connections) > self.current_size:
                    connection = connections.pop()
                    if hasattr(connection, 'close'):
                        if asyncio.iscoroutinefunction(connection.close):
                            await connection.close()
                        else:
                            connection.close()
    
    def get_pool_metrics(self) -> Dict[str, Any]:
        """Get pool performance metrics."""
        with self._lock:
            hit_rate = (self.pool_stats['pool_hits'] / 
                       max(1, self.pool_stats['pool_hits'] + self.pool_stats['pool_misses']))
            
            return {
                'current_size': self.current_size,
                'total_connections': self.pool_stats['total_connections'],
                'active_connections': self.pool_stats['active_connections'],
                'hit_rate': hit_rate,
                'recent_load': list(self.load_history)[-10:],
                'expansions': self.pool_stats['expansions'],
                'contractions': self.pool_stats['contractions']
            }


class MemoryManager:
    """Advanced memory management with monitoring and optimization."""
    
    def __init__(self, memory_limit_mb: int = 512):
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        self.caches: List[BoundedCache] = []
        self.memory_stats = {
            'peak_usage': 0,
            'gc_collections': 0,
            'cache_cleanups': 0,
            'memory_warnings': 0
        }
        
        self._monitoring_task: Optional[asyncio.Task] = None
        self._weak_refs: weakref.WeakSet = weakref.WeakSet()
    
    def register_cache(self, cache: BoundedCache) -> None:
        """Register cache for memory management."""
        self.caches.append(cache)
        self._weak_refs.add(cache)
    
    async def start_monitoring(self) -> None:
        """Start memory monitoring task."""
        if self._monitoring_task is None:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self) -> None:
        """Stop memory monitoring task."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
    
    async def _monitoring_loop(self) -> None:
        """Background memory monitoring loop."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._check_memory_usage()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
    
    async def _check_memory_usage(self) -> None:
        """Check and manage memory usage."""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            current_usage = memory_info.rss
            
            self.memory_stats['peak_usage'] = max(self.memory_stats['peak_usage'], 
                                                 current_usage)
            
            usage_ratio = current_usage / self.memory_limit_bytes
            
            if usage_ratio > 0.8:  # 80% threshold
                logger.warning(f"High memory usage: {usage_ratio:.1%}")
                self.memory_stats['memory_warnings'] += 1
                await self._cleanup_memory()
            
            elif usage_ratio > 0.9:  # 90% threshold - aggressive cleanup
                logger.error(f"Critical memory usage: {usage_ratio:.1%}")
                await self._aggressive_cleanup()
                
        except Exception as e:
            logger.error(f"Memory check failed: {e}")
    
    async def _cleanup_memory(self) -> None:
        """Perform memory cleanup."""
        # Clean up caches
        total_cleaned = 0
        for cache in list(self.caches):  # Copy list to avoid modification during iteration
            if cache in self._weak_refs:  # Check if still alive
                cleaned = cache.cleanup_expired()
                total_cleaned += cleaned
        
        if total_cleaned > 0:
            logger.info(f"Cleaned {total_cleaned} expired cache entries")
            self.memory_stats['cache_cleanups'] += 1
        
        # Force garbage collection
        collected = gc.collect()
        if collected > 0:
            logger.info(f"Garbage collected {collected} objects")
            self.memory_stats['gc_collections'] += 1
    
    async def _aggressive_cleanup(self) -> None:
        """Perform aggressive memory cleanup."""
        # Clear half of each cache
        for cache in list(self.caches):
            if cache in self._weak_refs:
                # Clear half the cache
                target_size = cache.max_size // 2
                while len(cache.cache) > target_size and cache.insertion_order:
                    cache._evict_lru()
        
        # Multiple GC passes
        for _ in range(3):
            gc.collect()
        
        logger.warning("Performed aggressive memory cleanup")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory management statistics."""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            return {
                'current_usage_mb': memory_info.rss / 1024 / 1024,
                'peak_usage_mb': self.memory_stats['peak_usage'] / 1024 / 1024,
                'limit_mb': self.memory_limit_bytes / 1024 / 1024,
                'usage_ratio': memory_info.rss / self.memory_limit_bytes,
                'cache_count': len(self.caches),
                'gc_collections': self.memory_stats['gc_collections'],
                'cache_cleanups': self.memory_stats['cache_cleanups'],
                'memory_warnings': self.memory_stats['memory_warnings']
            }
        except Exception:
            return {
                'current_usage_mb': 0,
                'peak_usage_mb': 0,
                'error': 'Unable to get memory stats'
            }


class OptimizedEventLoop:
    """Event loop optimizations for high performance."""
    
    def __init__(self):
        self.batch_size = 50
        self.batch_delay = 0.001  # 1ms
        self.pending_tasks: deque = deque()
        self.task_stats = {
            'batches_processed': 0,
            'tasks_processed': 0,
            'avg_batch_size': 0.0
        }
        
        # Try to use uvloop for better performance
        try:
            import uvloop
            if not isinstance(asyncio.get_event_loop(), uvloop.Loop):
                uvloop.install()
                logger.info("Installed uvloop for enhanced performance")
        except ImportError:
            logger.debug("uvloop not available, using default event loop")
    
    async def batch_execute(self, tasks: List[Callable]) -> List[Any]:
        """Execute tasks in optimized batches."""
        if not tasks:
            return []
        
        results = []
        
        # Process tasks in batches
        for i in range(0, len(tasks), self.batch_size):
            batch = tasks[i:i + self.batch_size]
            
            # Convert to coroutines if needed
            coros = []
            for task in batch:
                if asyncio.iscoroutinefunction(task):
                    coros.append(task())
                elif asyncio.iscoroutine(task):
                    coros.append(task)
                else:
                    # Wrap in coroutine
                    async def wrapped():
                        return task()
                    coros.append(wrapped())
            
            # Execute batch
            batch_results = await asyncio.gather(*coros, return_exceptions=True)
            results.extend(batch_results)
            
            # Update stats
            self.task_stats['batches_processed'] += 1
            self.task_stats['tasks_processed'] += len(batch)
            
            # Small delay between batches to prevent overwhelming
            if i + self.batch_size < len(tasks):
                await asyncio.sleep(self.batch_delay)
        
        # Update average batch size
        if self.task_stats['batches_processed'] > 0:
            self.task_stats['avg_batch_size'] = (
                self.task_stats['tasks_processed'] / 
                self.task_stats['batches_processed']
            )
        
        return results
    
    def get_event_loop_stats(self) -> Dict[str, Any]:
        """Get event loop performance statistics."""
        loop = asyncio.get_running_loop()
        
        return {
            'loop_type': type(loop).__name__,
            'batches_processed': self.task_stats['batches_processed'],
            'tasks_processed': self.task_stats['tasks_processed'],
            'avg_batch_size': self.task_stats['avg_batch_size'],
            'pending_tasks': len(self.pending_tasks)
        }


class PerformanceOrchestrator:
    """Main performance orchestrator combining all optimizations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        self.connection_pool = AdaptiveConnectionPool(
            base_size=config.get('pool_base_size', 10),
            max_size=config.get('pool_max_size', 100),
            min_size=config.get('pool_min_size', 5)
        )
        
        self.memory_manager = MemoryManager(
            memory_limit_mb=config.get('memory_limit_mb', 512)
        )
        
        self.event_loop = OptimizedEventLoop()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Performance caches
        self.route_cache = BoundedCache[str](max_size=1000, ttl=300)
        self.peer_cache = BoundedCache[Dict](max_size=500, ttl=600)
        
        # Register caches with memory manager
        self.memory_manager.register_cache(self.route_cache)
        self.memory_manager.register_cache(self.peer_cache)
        
        # Global metrics
        self.global_metrics = PerformanceMetrics()
    
    async def initialize(self) -> None:
        """Initialize performance orchestrator."""
        await self.connection_pool.start_adaptive_sizing()
        await self.memory_manager.start_monitoring()
        logger.info("Performance orchestrator initialized")
    
    async def shutdown(self) -> None:
        """Shutdown performance orchestrator."""
        await self.connection_pool.stop_adaptive_sizing()
        await self.memory_manager.stop_monitoring()
        logger.info("Performance orchestrator shut down")
    
    def get_circuit_breaker(self, service_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for service."""
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreaker(
                failure_threshold=self.config.get('cb_failure_threshold', 5),
                timeout=self.config.get('cb_timeout', 60),
                success_threshold=self.config.get('cb_success_threshold', 3)
            )
        return self.circuit_breakers[service_name]
    
    async def execute_with_circuit_breaker(self, service_name: str, 
                                         func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        cb = self.get_circuit_breaker(service_name)
        
        start_time = time.perf_counter()
        error = False
        
        try:
            result = await cb.call(func, *args, **kwargs)
            return result
        except Exception as e:
            error = True
            raise
        finally:
            latency = time.perf_counter() - start_time
            self.global_metrics.add_operation(latency, error)
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        return {
            'global_metrics': {
                'operations_per_second': self.global_metrics.operations_per_second,
                'average_latency': self.global_metrics.average_latency,
                'error_rate': self.global_metrics.error_rate,
                'total_operations': self.global_metrics.operations_count
            },
            'connection_pool': self.connection_pool.get_pool_metrics(),
            'memory_management': self.memory_manager.get_memory_stats(),
            'event_loop': self.event_loop.get_event_loop_stats(),
            'circuit_breakers': {
                name: {
                    'state': cb.state.value,
                    'failure_count': cb.failure_count,
                    'success_count': cb.success_count
                }
                for name, cb in self.circuit_breakers.items()
            },
            'caches': {
                'route_cache': self.route_cache.get_stats(),
                'peer_cache': self.peer_cache.get_stats()
            }
        }


# Example usage
async def example_performance_optimization():
    """Example of using performance optimizations."""
    config = {
        'pool_base_size': 10,
        'pool_max_size': 50,
        'memory_limit_mb': 256,
        'cb_failure_threshold': 3,
        'cb_timeout': 30
    }
    
    orchestrator = PerformanceOrchestrator(config)
    await orchestrator.initialize()
    
    try:
        # Example circuit breaker usage
        async def risky_operation():
            """Simulated risky operation."""
            if time.time() % 10 < 5:  # Fail 50% of the time
                raise Exception("Simulated failure")
            return "Success"
        
        # Execute with protection
        try:
            result = await orchestrator.execute_with_circuit_breaker(
                "test_service", risky_operation
            )
            print(f"Operation result: {result}")
        except CircuitBreakerOpen:
            print("Circuit breaker is open, skipping operation")
        except Exception as e:
            print(f"Operation failed: {e}")
        
        # Get metrics
        metrics = orchestrator.get_comprehensive_metrics()
        print(f"Performance metrics: {metrics}")
        
    finally:
        await orchestrator.shutdown()


if __name__ == "__main__":
    asyncio.run(example_performance_optimization())
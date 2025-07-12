"""
Lazy Import System
=================

Provides lazy loading of heavy modules to improve startup time.
"""

import importlib
import logging
from typing import Any, Dict, List, Optional, Union
from functools import lru_cache

logger = logging.getLogger(__name__)


class LazyModule:
    """Lazy loader for a single module."""
    
    def __init__(self, module_name: str):
        self.module_name = module_name
        self._module = None
        self._loaded = False
        self._error = None
    
    def _load(self):
        """Load the module if not already loaded."""
        if self._loaded:
            return
        
        try:
            self._module = importlib.import_module(self.module_name)
            self._loaded = True
            logger.debug(f"✅ Lazy loaded: {self.module_name}")
        except ImportError as e:
            self._error = e
            self._loaded = True
            logger.warning(f"❌ Failed to lazy load {self.module_name}: {e}")
    
    def __getattr__(self, name: str):
        """Get attribute from the lazily loaded module."""
        self._load()
        
        if self._error:
            raise ImportError(f"Module {self.module_name} failed to load: {self._error}")
        
        if self._module is None:
            raise ImportError(f"Module {self.module_name} is not available")
        
        return getattr(self._module, name)
    
    def __bool__(self):
        """Check if module is available."""
        self._load()
        return self._error is None and self._module is not None
    
    @property
    def is_loaded(self) -> bool:
        """Check if module has been loaded."""
        return self._loaded
    
    @property
    def is_available(self) -> bool:
        """Check if module is available without loading."""
        if not self._loaded:
            self._load()
        return self._error is None


class LazyImporter:
    """Manages lazy loading of multiple modules."""
    
    def __init__(self, modules: Union[str, List[str]]):
        if isinstance(modules, str):
            modules = [modules]
        
        self.modules: Dict[str, LazyModule] = {}
        for module_name in modules:
            self.modules[module_name] = LazyModule(module_name)
    
    def get_module(self, module_name: str) -> LazyModule:
        """Get a lazy module by name."""
        if module_name not in self.modules:
            self.modules[module_name] = LazyModule(module_name)
        return self.modules[module_name]
    
    def get_class(self, module_name: str, class_name: str) -> Optional[type]:
        """Get a class from a lazily loaded module."""
        try:
            module = self.get_module(module_name)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            logger.warning(f"Failed to get {module_name}.{class_name}: {e}")
            return None
    
    def is_available(self, module_name: str) -> bool:
        """Check if a module is available."""
        return self.get_module(module_name).is_available
    
    def load_all(self) -> Dict[str, bool]:
        """Load all modules and return availability status."""
        status = {}
        for module_name, lazy_module in self.modules.items():
            lazy_module._load()
            status[module_name] = lazy_module.is_available
        return status
    
    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed status of all modules."""
        status = {}
        for module_name, lazy_module in self.modules.items():
            status[module_name] = {
                'loaded': lazy_module.is_loaded,
                'available': lazy_module.is_available,
                'error': str(lazy_module._error) if lazy_module._error else None
            }
        return status


class LazyFunction:
    """Lazy loader for specific functions from modules."""
    
    def __init__(self, module_name: str, function_name: str):
        self.module_name = module_name
        self.function_name = function_name
        self._function = None
        self._loaded = False
        self._error = None
    
    def _load(self):
        """Load the function if not already loaded."""
        if self._loaded:
            return
        
        try:
            module = importlib.import_module(self.module_name)
            self._function = getattr(module, self.function_name)
            self._loaded = True
            logger.debug(f"✅ Lazy loaded function: {self.module_name}.{self.function_name}")
        except (ImportError, AttributeError) as e:
            self._error = e
            self._loaded = True
            logger.warning(f"❌ Failed to lazy load {self.module_name}.{self.function_name}: {e}")
    
    def __call__(self, *args, **kwargs):
        """Call the lazily loaded function."""
        self._load()
        
        if self._error:
            raise ImportError(f"Function {self.module_name}.{self.function_name} failed to load: {self._error}")
        
        if self._function is None:
            raise ImportError(f"Function {self.module_name}.{self.function_name} is not available")
        
        return self._function(*args, **kwargs)
    
    @property
    def is_available(self) -> bool:
        """Check if function is available."""
        if not self._loaded:
            self._load()
        return self._error is None


@lru_cache(maxsize=128)
def create_lazy_module(module_name: str) -> LazyModule:
    """Create a cached lazy module."""
    return LazyModule(module_name)


@lru_cache(maxsize=128)
def create_lazy_function(module_name: str, function_name: str) -> LazyFunction:
    """Create a cached lazy function."""
    return LazyFunction(module_name, function_name)


# Predefined lazy importers for common heavy modules
AI_MODULES = LazyImporter([
    'transformers',
    'torch',
    'tensorflow',
    'langchain',
    'openai'
])

QUANTUM_MODULES = LazyImporter([
    'qiskit',
    'cirq',
    'pennylane'
])

MONITORING_MODULES = LazyImporter([
    'prometheus_client',
    'opentelemetry.api',
    'jaeger_client'
])

DATA_MODULES = LazyImporter([
    'pandas',
    'numpy',
    'scipy',
    'matplotlib',
    'plotly'
])

WEB_MODULES = LazyImporter([
    'dash',
    'streamlit',
    'gradio',
    'bokeh'
])


def get_predefined_importer(category: str) -> Optional[LazyImporter]:
    """Get a predefined lazy importer by category."""
    importers = {
        'ai': AI_MODULES,
        'quantum': QUANTUM_MODULES,
        'monitoring': MONITORING_MODULES,
        'data': DATA_MODULES,
        'web': WEB_MODULES
    }
    return importers.get(category.lower())


def lazy_import_decorator(module_name: str):
    """Decorator for lazy importing in functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                module = importlib.import_module(module_name)
                # Add module to function's globals
                func.__globals__[module_name.split('.')[-1]] = module
                return func(*args, **kwargs)
            except ImportError as e:
                raise ImportError(f"Required module {module_name} not available: {e}")
        return wrapper
    return decorator


# Example usage and testing
if __name__ == "__main__":
    import time
    
    print("Testing lazy imports...")
    
    # Test lazy module
    print("\n1. Testing LazyModule:")
    lazy_json = LazyModule('json')
    print(f"Available: {lazy_json.is_available}")
    print(f"Loads: {lazy_json.loads('{}')}")
    
    # Test lazy importer
    print("\n2. Testing LazyImporter:")
    importer = LazyImporter(['json', 'sys', 'nonexistent_module'])
    status = importer.get_status()
    for module, info in status.items():
        print(f"  {module}: {info}")
    
    # Test lazy function
    print("\n3. Testing LazyFunction:")
    lazy_dumps = LazyFunction('json', 'dumps')
    print(f"Result: {lazy_dumps({'test': 'data'})}")
    
    # Test performance
    print("\n4. Performance test:")
    start = time.time()
    heavy_modules = LazyImporter(['collections', 'itertools', 'functools'])
    creation_time = time.time() - start
    print(f"Creation time: {creation_time:.4f}s")
    
    start = time.time()
    heavy_modules.load_all()
    load_time = time.time() - start
    print(f"Load time: {load_time:.4f}s")
    
    # Test predefined importers
    print("\n5. Predefined importers:")
    ai_importer = get_predefined_importer('ai')
    if ai_importer:
        ai_status = ai_importer.get_status()
        for module, info in ai_status.items():
            status = "✅" if info['available'] else "❌"
            print(f"  {status} {module}")
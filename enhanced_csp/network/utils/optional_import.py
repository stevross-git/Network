"""
Optional Import Utilities
========================

Provides graceful handling of optional dependencies with consistent patterns.
"""

import logging
from typing import Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def optional_import(
    module_name: str, 
    package: Optional[str] = None,
    min_version: Optional[str] = None
) -> Tuple[Any, bool]:
    """
    Import module with graceful fallback.
    
    Args:
        module_name: Name of the module to import
        package: Specific package/class to import from module
        min_version: Minimum version required
        
    Returns:
        Tuple of (module/class, is_available)
    """
    try:
        if package:
            module = __import__(module_name, fromlist=[package])
            imported = getattr(module, package)
        else:
            imported = __import__(module_name)
        
        # Version checking if required
        if min_version and hasattr(imported, '__version__'):
            from packaging import version
            if version.parse(imported.__version__) < version.parse(min_version):
                logger.warning(
                    f"{module_name} version {imported.__version__} < {min_version}, "
                    "some features may not work correctly"
                )
        
        logger.debug(f"✅ {module_name} imported successfully")
        return imported, True
        
    except ImportError as e:
        logger.warning(f"❌ {module_name} not available: {e}")
        return None, False
    except AttributeError as e:
        logger.warning(f"❌ {module_name}.{package} not found: {e}")
        return None, False


def batch_optional_imports(imports: dict) -> dict:
    """
    Import multiple optional modules at once.
    
    Args:
        imports: Dict of {name: (module, package, min_version)}
        
    Returns:
        Dict of {name: (imported_object, is_available)}
    """
    results = {}
    
    for name, import_spec in imports.items():
        if isinstance(import_spec, str):
            # Simple module name
            results[name] = optional_import(import_spec)
        elif isinstance(import_spec, tuple):
            # (module, package, min_version)
            if len(import_spec) == 2:
                module, package = import_spec
                results[name] = optional_import(module, package)
            elif len(import_spec) == 3:
                module, package, min_version = import_spec
                results[name] = optional_import(module, package, min_version)
        else:
            logger.error(f"Invalid import spec for {name}: {import_spec}")
            results[name] = (None, False)
    
    return results


def check_dependencies(dependencies: list) -> dict:
    """
    Check availability of multiple dependencies.
    
    Args:
        dependencies: List of module names or tuples
        
    Returns:
        Dict with availability status and versions
    """
    status = {
        'available': [],
        'missing': [],
        'versions': {}
    }
    
    for dep in dependencies:
        if isinstance(dep, str):
            module, available = optional_import(dep)
        else:
            module, available = optional_import(*dep)
        
        dep_name = dep if isinstance(dep, str) else dep[0]
        
        if available:
            status['available'].append(dep_name)
            if hasattr(module, '__version__'):
                status['versions'][dep_name] = module.__version__
        else:
            status['missing'].append(dep_name)
    
    return status


def create_fallback_module(module_name: str, fallback_classes: dict):
    """
    Create a fallback module with stub classes.
    
    Args:
        module_name: Name for the fallback module
        fallback_classes: Dict of {class_name: fallback_class}
    """
    import types
    
    fallback_module = types.ModuleType(module_name)
    
    for class_name, fallback_class in fallback_classes.items():
        setattr(fallback_module, class_name, fallback_class)
    
    return fallback_module


# Predefined import configurations for common dependencies
COMMON_IMPORTS = {
    'redis': ('redis.asyncio', None, '4.0.0'),
    'prometheus': ('prometheus_client', None, '0.8.0'),
    'psutil': ('psutil', None, '5.0.0'),
    'aiofiles': ('aiofiles', None, '0.8.0'),
    'aiohttp': ('aiohttp', None, '3.8.0'),
    'uvloop': ('uvloop', None, '0.15.0'),
    'orjson': ('orjson', None, '3.0.0'),
    'msgpack': ('msgpack', None, '1.0.0'),
    'lz4': ('lz4', None, '3.0.0'),
    'zstandard': ('zstandard', None, '0.15.0'),
}


def load_common_imports() -> dict:
    """Load all common optional imports."""
    return batch_optional_imports(COMMON_IMPORTS)


# Example usage and testing
if __name__ == "__main__":
    import sys
    
    print("Testing optional imports...")
    
    # Test individual import
    redis, redis_available = optional_import('redis.asyncio')
    print(f"Redis: {redis_available}")
    
    # Test batch imports
    results = load_common_imports()
    print("\nCommon imports:")
    for name, (module, available) in results.items():
        status = "✅" if available else "❌"
        print(f"  {status} {name}")
    
    # Test dependency checking
    deps = ['fastapi', 'uvicorn', 'nonexistent_module']
    status = check_dependencies(deps)
    print(f"\nDependency check:")
    print(f"Available: {status['available']}")
    print(f"Missing: {status['missing']}")
    print(f"Versions: {status['versions']}")
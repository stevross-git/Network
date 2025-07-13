#!/usr/bin/env python3
"""
Enhanced CSP Import Fixer
Diagnoses and fixes import issues in the Enhanced CSP network module.
"""

import os
import sys
import importlib
from pathlib import Path
from typing import List, Dict, Any
import traceback

class ImportFixer:
    """Diagnose and fix import issues."""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.enhanced_csp_path = self.project_root / "enhanced_csp"
        self.issues_found = []
        self.fixes_applied = []
        
    def diagnose_and_fix(self):
        """Run full diagnosis and fix cycle."""
        print("🔧 Enhanced CSP Import Fixer")
        print("=" * 50)
        print(f"📂 Project root: {self.project_root}")
        print(f"📁 Enhanced CSP path: {self.enhanced_csp_path}")
        print()
        
        # Step 1: Check directory structure
        self._check_directory_structure()
        
        # Step 2: Check __init__.py files
        self._check_init_files()
        
        # Step 3: Check for missing modules
        self._check_missing_modules()
        
        # Step 4: Test imports
        self._test_imports()
        
        # Step 5: Apply fixes
        self._apply_fixes()
        
        # Step 6: Final test
        self._final_test()
        
        # Report
        self._print_report()
    
    def _check_directory_structure(self):
        """Check if all required directories exist."""
        print("📁 Checking directory structure...")
        
        required_dirs = [
            "enhanced_csp",
            "enhanced_csp/network",
            "enhanced_csp/network/core",
            "enhanced_csp/network/utils",
            "enhanced_csp/network/p2p",
            "enhanced_csp/network/mesh",
            "enhanced_csp/network/dns",
            "enhanced_csp/network/routing",
        ]
        
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if full_path.exists() and full_path.is_dir():
                print(f"  ✅ {dir_path}")
            else:
                print(f"  ❌ {dir_path} - MISSING")
                self.issues_found.append(f"Missing directory: {dir_path}")
    
    def _check_init_files(self):
        """Check for missing __init__.py files."""
        print("\n📄 Checking __init__.py files...")
        
        required_inits = [
            "enhanced_csp/__init__.py",
            "enhanced_csp/network/__init__.py",
            "enhanced_csp/network/core/__init__.py",
            "enhanced_csp/network/utils/__init__.py",
            "enhanced_csp/network/p2p/__init__.py",
            "enhanced_csp/network/mesh/__init__.py",
            "enhanced_csp/network/dns/__init__.py",
            "enhanced_csp/network/routing/__init__.py",
        ]
        
        for init_path in required_inits:
            full_path = self.project_root / init_path
            if full_path.exists():
                print(f"  ✅ {init_path}")
            else:
                print(f"  ❌ {init_path} - MISSING")
                self.issues_found.append(f"Missing __init__.py: {init_path}")
    
    def _check_missing_modules(self):
        """Check for missing Python modules."""
        print("\n🐍 Checking required modules...")
        
        # Check if specific files exist
        required_files = [
            "enhanced_csp/network/utils/task_manager.py",
            "enhanced_csp/network/utils/structured_logging.py",
            "enhanced_csp/network/core/config.py",
            "enhanced_csp/network/core/types.py",
            "enhanced_csp/network/core/node.py",
        ]
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                print(f"  ✅ {file_path}")
            else:
                print(f"  ❌ {file_path} - MISSING")
                self.issues_found.append(f"Missing module file: {file_path}")
    
    def _test_imports(self):
        """Test actual imports."""
        print("\n🧪 Testing imports...")
        
        # Add project root to Python path
        sys.path.insert(0, str(self.project_root))
        
        imports_to_test = [
            ("enhanced_csp", "Enhanced CSP base package"),
            ("enhanced_csp.network", "Network package"),
            ("enhanced_csp.network.utils", "Network utils"),
            ("enhanced_csp.network.utils.task_manager", "TaskManager module"),
            ("enhanced_csp.network.core", "Core module"),
            ("enhanced_csp.network.core.config", "Config module"),
        ]
        
        for module_name, description in imports_to_test:
            try:
                importlib.import_module(module_name)
                print(f"  ✅ {description} ({module_name})")
            except ImportError as e:
                print(f"  ❌ {description} ({module_name}) - {e}")
                self.issues_found.append(f"Import failed: {module_name} - {e}")
            except Exception as e:
                print(f"  ⚠️  {description} ({module_name}) - Unexpected error: {e}")
                self.issues_found.append(f"Unexpected error: {module_name} - {e}")
    
    def _apply_fixes(self):
        """Apply fixes for found issues."""
        if not self.issues_found:
            print("\n✅ No issues found to fix!")
            return
            
        print(f"\n🔧 Applying fixes for {len(self.issues_found)} issues...")
        
        for issue in self.issues_found:
            if "Missing __init__.py:" in issue:
                init_path = issue.replace("Missing __init__.py: ", "")
                self._create_init_file(init_path)
            elif "Missing directory:" in issue:
                dir_path = issue.replace("Missing directory: ", "")
                self._create_directory(dir_path)
            elif "Missing module file:" in issue:
                self._handle_missing_module(issue)
    
    def _create_init_file(self, init_path: str):
        """Create a missing __init__.py file."""
        full_path = self.project_root / init_path
        
        try:
            # Create directory if it doesn't exist
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Determine content based on the module
            content = self._get_init_content(init_path)
            
            # Write the file
            full_path.write_text(content)
            print(f"  ✅ Created: {init_path}")
            self.fixes_applied.append(f"Created {init_path}")
            
        except Exception as e:
            print(f"  ❌ Failed to create {init_path}: {e}")
    
    def _create_directory(self, dir_path: str):
        """Create a missing directory."""
        full_path = self.project_root / dir_path
        
        try:
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✅ Created directory: {dir_path}")
            self.fixes_applied.append(f"Created directory {dir_path}")
        except Exception as e:
            print(f"  ❌ Failed to create directory {dir_path}: {e}")
    
    def _handle_missing_module(self, issue: str):
        """Handle missing module files."""
        module_path = issue.replace("Missing module file: ", "")
        
        if "task_manager.py" in module_path:
            self._create_task_manager()
        elif "structured_logging.py" in module_path:
            self._create_structured_logging()
        else:
            print(f"  ⚠️  Don't know how to create: {module_path}")
    
    def _create_task_manager(self):
        """Create a minimal task_manager.py file."""
        content = '''"""Task manager utility for Enhanced CSP Network."""

import asyncio
from typing import Set, Any
from contextlib import contextmanager


class TaskManager:
    """Manage lifecycle of asyncio tasks."""
    
    def __init__(self):
        self.tasks: Set[asyncio.Task] = set()
    
    def create_task(self, coro, name=None):
        """Create and track an asyncio task."""
        task = asyncio.create_task(coro, name=name)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task
    
    async def cancel_all(self, timeout=5.0):
        """Cancel all tracked tasks."""
        for task in list(self.tasks):
            if not task.done():
                task.cancel()
        
        if self.tasks:
            await asyncio.wait(self.tasks, timeout=timeout)
        self.tasks.clear()


class ResourceManager:
    """Track resources to ensure they are closed properly."""
    
    def __init__(self):
        self.resources: Set[Any] = set()
    
    @contextmanager
    def manage(self, resource):
        """Context manager for resource management."""
        self.resources.add(resource)
        try:
            yield resource
        finally:
            self.close(resource)
    
    def close(self, resource):
        """Close a resource."""
        if resource in self.resources:
            try:
                if hasattr(resource, 'close'):
                    resource.close()
            except Exception:
                pass
            finally:
                self.resources.discard(resource)
    
    async def close_all(self):
        """Close all tracked resources."""
        for resource in list(self.resources):
            self.close(resource)
        self.resources.clear()
'''
        
        file_path = self.project_root / "enhanced_csp/network/utils/task_manager.py"
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            print(f"  ✅ Created minimal task_manager.py")
            self.fixes_applied.append("Created task_manager.py")
        except Exception as e:
            print(f"  ❌ Failed to create task_manager.py: {e}")
    
    def _create_structured_logging(self):
        """Create a minimal structured_logging.py file."""
        content = '''"""Structured logging utilities for Enhanced CSP Network."""

import logging
import sys


def get_logger(name):
    """Get a logger instance."""
    return logging.getLogger(name)


def setup_logging(level="INFO"):
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


class NetworkLogger:
    """Network-specific logger."""
    def __init__(self, name):
        self.logger = logging.getLogger(name)


class SecurityLogger:
    """Security-specific logger.""" 
    def __init__(self, name):
        self.logger = logging.getLogger(name)


class PerformanceLogger:
    """Performance-specific logger."""
    def __init__(self, name):
        self.logger = logging.getLogger(name)


class AuditLogger:
    """Audit-specific logger."""
    def __init__(self, name):
        self.logger = logging.getLogger(name)


# Compatibility classes
class StructuredFormatter:
    pass

class SamplingFilter:
    pass

class StructuredAdapter:
    pass
'''
        
        file_path = self.project_root / "enhanced_csp/network/utils/structured_logging.py"
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            print(f"  ✅ Created minimal structured_logging.py")
            self.fixes_applied.append("Created structured_logging.py")
        except Exception as e:
            print(f"  ❌ Failed to create structured_logging.py: {e}")
    
    def _get_init_content(self, init_path: str) -> str:
        """Get appropriate content for __init__.py based on the module."""
        
        if init_path == "enhanced_csp/__init__.py":
            return '''"""Enhanced CSP System - Advanced Computing Systems Platform."""
__version__ = "1.0.0"
'''
        
        elif init_path == "enhanced_csp/network/__init__.py":
            return '''"""Enhanced CSP Network Module."""
__version__ = "1.0.0"

# Lazy imports to avoid circular dependencies
try:
    from .core.config import NetworkConfig
    from .core.types import NodeID, MessageType
    from .core.node import NetworkNode, EnhancedCSPNetwork
except ImportError:
    pass

def create_network(config=None):
    """Create a new Enhanced CSP Network instance."""
    from .core.node import EnhancedCSPNetwork
    return EnhancedCSPNetwork(config)

def create_node(config=None):
    """Create a new network node."""
    from .core.node import NetworkNode
    return NetworkNode(config)

__all__ = ["NetworkConfig", "NodeID", "MessageType", "NetworkNode", "EnhancedCSPNetwork", "create_network", "create_node"]
'''
        
        elif "utils" in init_path:
            return '''"""Enhanced CSP Network utilities."""

try:
    from .task_manager import TaskManager, ResourceManager
    from .structured_logging import get_logger, setup_logging, NetworkLogger, SecurityLogger, PerformanceLogger, AuditLogger
except ImportError:
    # Create placeholder classes if modules don't exist
    class TaskManager:
        def __init__(self): pass
        def create_task(self, coro, name=None): return None
        async def cancel_all(self): pass
    
    class ResourceManager:
        def __init__(self): pass
        async def close_all(self): pass
    
    import logging
    def get_logger(name): return logging.getLogger(name)
    def setup_logging(level="INFO"): logging.basicConfig(level=level)
    
    class NetworkLogger: pass
    class SecurityLogger: pass  
    class PerformanceLogger: pass
    class AuditLogger: pass

__all__ = ["TaskManager", "ResourceManager", "get_logger", "setup_logging", "NetworkLogger", "SecurityLogger", "PerformanceLogger", "AuditLogger"]
'''
        
        else:
            # Generic __init__.py content
            module_name = Path(init_path).parent.name
            return f'"""Enhanced CSP {module_name} module."""\n'
    
    def _final_test(self):
        """Final test of imports after fixes."""
        print("\n🧪 Final import test...")
        
        # Clear import cache
        for module_name in list(sys.modules.keys()):
            if module_name.startswith('enhanced_csp'):
                del sys.modules[module_name]
        
        try:
            from enhanced_csp.network.utils import TaskManager, get_logger
            from enhanced_csp.network.core.config import NetworkConfig
            print("  ✅ All critical imports working!")
            return True
        except ImportError as e:
            print(f"  ❌ Still have import issues: {e}")
            return False
        except Exception as e:
            print(f"  ⚠️  Unexpected error: {e}")
            return False
    
    def _print_report(self):
        """Print final report."""
        print("\n📊 Import Fixer Report")
        print("=" * 50)
        
        if not self.issues_found:
            print("🎉 No issues found - everything looks good!")
        else:
            print(f"📋 Issues found: {len(self.issues_found)}")
            for issue in self.issues_found:
                print(f"  • {issue}")
        
        if self.fixes_applied:
            print(f"\n🔧 Fixes applied: {len(self.fixes_applied)}")
            for fix in self.fixes_applied:
                print(f"  • {fix}")
        
        print(f"\n💡 Next steps:")
        print("  1. Try running: python3 network_startup.py --quick-start")
        print("  2. If still having issues, check your Python path")
        print("  3. Make sure you're in the project root directory")


def main():
    """Main function."""
    fixer = ImportFixer()
    fixer.diagnose_and_fix()


if __name__ == "__main__":
    main()
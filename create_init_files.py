#!/usr/bin/env python3
"""
Enhanced CSP Network - __init__.py File Creator
Automatically creates missing __init__.py files with appropriate content.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InitFileCreator:
    """Creates __init__.py files with appropriate content for Python packages."""
    
    def __init__(self, project_root: Path = None):
        """Initialize the creator with project root directory."""
        if project_root is None:
            # Auto-detect project root
            current_file = Path(__file__).resolve()
            self.project_root = self._find_project_root(current_file)
        else:
            self.project_root = Path(project_root).resolve()
        
        logger.info(f"🔍 Project root detected: {self.project_root}")
        
        # Define package structure and their init file contents
        self.package_configs = self._get_package_configurations()
        
        # Track created files
        self.created_files: List[Path] = []
        self.existing_files: List[Path] = []
        self.failed_files: List[Tuple[Path, str]] = []
    
    def _find_project_root(self, start_path: Path) -> Path:
        """Find the project root by looking for enhanced_csp directory."""
        current = start_path.parent if start_path.is_file() else start_path
        
        # Look for enhanced_csp directory or other project indicators
        for parent in [current] + list(current.parents):
            if (parent / "enhanced_csp").exists():
                return parent
            if (parent / "setup.py").exists() or (parent / "pyproject.toml").exists():
                return parent
        
        # Default to current directory if not found
        logger.warning("Could not auto-detect project root, using current directory")
        return Path.cwd()
    
    def _get_package_configurations(self) -> Dict[str, Dict]:
        """Define the package structure and init file contents."""
        return {
            "enhanced_csp": {
                "content": '''"""Enhanced CSP System - Advanced Computing Systems Platform.

This package provides a comprehensive platform for distributed computing,
peer-to-peer networking, AI integration, and quantum computing capabilities.
"""

__version__ = "1.0.0"
__author__ = "Enhanced CSP Team"
__email__ = "team@enhanced-csp.com"
__description__ = "Advanced Computing Systems Platform"

# Package metadata
__all__ = [
    "network",
    "__version__",
    "__author__",
    "__email__",
    "__description__",
]
''',
                "priority": 1
            },
            
            "enhanced_csp/network": {
                "content": '''"""Enhanced CSP Network Module.

Provides peer-to-peer networking, mesh routing, peer discovery,
connection pooling, and performance optimizations.
"""

__version__ = "1.0.0"

# Import core components for easy access
try:
    from .core.config import NetworkConfig
    from .core.types import NodeID, PeerInfo, NetworkMessage
    __all__ = ["NetworkConfig", "NodeID", "PeerInfo", "NetworkMessage"]
except ImportError:
    # Graceful fallback if core modules aren't available yet
    __all__ = []

# Network module metadata
__description__ = "Enhanced CSP Network Layer"
''',
                "priority": 2
            },
            
            "enhanced_csp/network/core": {
                "content": '''"""Core networking components and types.

Contains fundamental types, configuration classes, and base components
used throughout the Enhanced CSP Network system.
"""

# Core exports
__all__ = [
    "config",
    "types", 
    "errors",
    "base",
]
''',
                "priority": 3
            },
            
            "enhanced_csp/network/utils": {
                "content": '''"""Network utility functions and helpers.

Provides common utilities for networking operations, data processing,
and system integration.
"""

__all__ = [
    "helpers",
    "validators",
    "converters",
]
''',
                "priority": 3
            },
            
            "enhanced_csp/network/security": {
                "content": '''"""Network security and cryptography modules.

Handles encryption, authentication, key exchange, and other
security-related functionality for the network layer.
"""

__all__ = [
    "encryption",
    "auth",
    "keys",
    "protocols",
]
''',
                "priority": 3
            },
            
            "enhanced_csp/network/p2p": {
                "content": '''"""Peer-to-peer networking implementation.

Provides DHT (Distributed Hash Table), NAT traversal, transport protocols,
and peer discovery mechanisms.
"""

__all__ = [
    "dht",
    "nat_traversal", 
    "transport",
    "discovery",
    "quic_transport",
]
''',
                "priority": 3
            },
            
            "enhanced_csp/network/mesh": {
                "content": '''"""Mesh network topology and management.

Implements mesh networking algorithms, topology optimization,
and network structure management.
"""

__all__ = [
    "topology",
    "routing",
    "optimization",
    "algorithms",
]
''',
                "priority": 3
            },
            
            "enhanced_csp/network/dns": {
                "content": '''"""DNS overlay and service discovery.

Provides lightweight DNS functionality for service discovery
and name resolution within the Enhanced CSP network.
"""

__all__ = [
    "resolver",
    "discovery",
    "cache",
]
''',
                "priority": 3
            },
            
            "enhanced_csp/network/routing": {
                "content": '''"""Advanced routing algorithms and engines.

Implements adaptive routing, multipath algorithms, and
intelligent traffic distribution mechanisms.
"""

__all__ = [
    "adaptive",
    "multipath",
    "algorithms",
    "engine",
]
''',
                "priority": 3
            },
            
            "enhanced_csp/network/optimization": {
                "content": '''"""Network performance optimization modules.

Contains batching, compression, zero-copy I/O, and other
performance enhancement components.
"""

__all__ = [
    "batching",
    "compression", 
    "zero_copy",
    "protocol_optimizer",
    "adaptive_optimizer",
]
''',
                "priority": 3
            },
            
            "enhanced_csp/network/examples": {
                "content": '''"""Example applications and demonstrations.

Contains example code showing how to use the Enhanced CSP Network
features and capabilities.
"""

__all__ = [
    "speed_optimization_example",
    "basic_node_example",
    "mesh_example",
]
''',
                "priority": 4
            },
            
            "enhanced_csp/network/dashboard": {
                "content": '''"""Real-time network monitoring dashboard.

Provides web-based monitoring, metrics collection, and
network visualization capabilities.
"""

__all__ = [
    "server",
    "metrics",
    "visualization",
]
''',
                "priority": 4
            },
            
            "enhanced_csp/network/tests": {
                "content": '''"""Test suite for the Enhanced CSP Network.

Contains unit tests, integration tests, and functional tests
for all network components.
"""

__all__ = [
    "test_core",
    "test_p2p",
    "test_mesh", 
    "test_routing",
    "test_optimization",
]
''',
                "priority": 5
            }
        }
    
    def check_existing_files(self) -> Dict[str, bool]:
        """Check which __init__.py files already exist."""
        status = {}
        
        for package_path in self.package_configs.keys():
            init_file = self.project_root / package_path / "__init__.py"
            exists = init_file.exists()
            status[package_path] = exists
            
            if exists:
                self.existing_files.append(init_file)
        
        return status
    
    def create_init_file(self, package_path: str, force: bool = False) -> bool:
        """Create a single __init__.py file."""
        config = self.package_configs[package_path]
        init_file = self.project_root / package_path / "__init__.py"
        
        # Check if file already exists
        if init_file.exists() and not force:
            logger.info(f"⏭️  Skipping existing file: {init_file}")
            return True
        
        try:
            # Create directory if it doesn't exist
            init_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write the init file content
            init_file.write_text(config["content"], encoding="utf-8")
            
            self.created_files.append(init_file)
            logger.info(f"✅ Created: {init_file}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to create {init_file}: {e}"
            self.failed_files.append((init_file, error_msg))
            logger.error(f"❌ {error_msg}")
            return False
    
    def create_all_init_files(self, force: bool = False) -> bool:
        """Create all missing __init__.py files."""
        logger.info("🚀 Starting __init__.py file creation process...")
        
        # Sort packages by priority
        sorted_packages = sorted(
            self.package_configs.items(),
            key=lambda x: x[1]["priority"]
        )
        
        success_count = 0
        total_count = len(sorted_packages)
        
        for package_path, config in sorted_packages:
            if self.create_init_file(package_path, force=force):
                success_count += 1
        
        # Print summary
        logger.info(f"\n📊 Summary:")
        logger.info(f"   ✅ Created: {len(self.created_files)} files")
        logger.info(f"   ⏭️  Existing: {len(self.existing_files)} files") 
        logger.info(f"   ❌ Failed: {len(self.failed_files)} files")
        logger.info(f"   📈 Success rate: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
        
        if self.failed_files:
            logger.error(f"\n❌ Failed files:")
            for file_path, error in self.failed_files:
                logger.error(f"   {file_path}: {error}")
        
        return len(self.failed_files) == 0
    
    def verify_imports(self) -> bool:
        """Verify that the created packages can be imported."""
        logger.info(f"\n🔍 Verifying imports...")
        
        # Add project root to Python path
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))
        
        test_imports = [
            "enhanced_csp",
            "enhanced_csp.network",
            "enhanced_csp.network.core",
            "enhanced_csp.network.utils",
            "enhanced_csp.network.p2p",
            "enhanced_csp.network.mesh",
        ]
        
        successful_imports = 0
        total_imports = len(test_imports)
        
        for import_name in test_imports:
            try:
                __import__(import_name)
                logger.info(f"   ✅ {import_name}")
                successful_imports += 1
            except ImportError as e:
                logger.error(f"   ❌ {import_name}: {e}")
        
        success_rate = successful_imports / total_imports
        logger.info(f"   📈 Import success rate: {success_rate:.1%}")
        
        return success_rate >= 0.8  # 80% success rate threshold
    
    def print_file_tree(self):
        """Print the created file tree structure."""
        logger.info(f"\n🌳 Created file tree:")
        
        for file_path in sorted(self.created_files):
            relative_path = file_path.relative_to(self.project_root)
            level = len(relative_path.parts) - 1
            indent = "  " * level
            logger.info(f"   {indent}📄 {relative_path}")


def main():
    """Main function to create __init__.py files."""
    print("🔧 Enhanced CSP Network - __init__.py File Creator")
    print("=" * 55)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Create missing __init__.py files")
    parser.add_argument("--project-root", help="Project root directory")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument("--verify", action="store_true", help="Verify imports after creation")
    parser.add_argument("--tree", action="store_true", help="Show file tree")
    
    args = parser.parse_args()
    
    try:
        # Create the init file creator
        creator = InitFileCreator(project_root=args.project_root)
        
        # Check existing files first
        existing_status = creator.check_existing_files()
        logger.info(f"📋 Found {sum(existing_status.values())} existing __init__.py files")
        
        # Create all init files
        success = creator.create_all_init_files(force=args.force)
        
        # Show file tree if requested
        if args.tree:
            creator.print_file_tree()
        
        # Verify imports if requested
        if args.verify:
            import_success = creator.verify_imports()
            if not import_success:
                logger.warning("⚠️  Some imports failed - check for missing dependencies")
        
        if success:
            print(f"\n🎉 Successfully created all __init__.py files!")
            print(f"💡 Your Enhanced CSP Network package structure is now complete!")
        else:
            print(f"\n⚠️  Some files failed to create - check the logs above")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Process failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

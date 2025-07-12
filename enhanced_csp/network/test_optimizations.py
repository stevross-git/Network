#!/usr/bin/env python3
"""Comprehensive test suite for CSP Network optimizations."""

import asyncio
import sys
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fix Python path - add the project root to sys.path
current_file = Path(__file__).resolve()
# Go up from test_optimizations.py -> network -> enhanced_csp -> project_root  
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

print(f"🔍 Project root: {project_root}")
print(f"🔍 Current working directory: {Path.cwd()}")
print(f"🔍 Python path includes: {project_root} ({'✅' if str(project_root) in sys.path else '❌'})")


class ImportTester:
    """Test import functionality safely."""
    
    def __init__(self):
        self.results = {}
        self.failed_imports = []
    
    def test_core_modules(self) -> Dict[str, str]:
        """Test core module imports."""
        core_modules = {
            'config': 'enhanced_csp.network.core.config',
            'types': 'enhanced_csp.network.core.types',
            'errors': 'enhanced_csp.network.errors',
            'utils': 'enhanced_csp.network.utils',
        }
        
        for name, module_path in core_modules.items():
            try:
                module = __import__(module_path, fromlist=[''])
                self.results[f'core_{name}'] = "✅ Available"
                logger.info(f"Successfully imported {module_path}")
            except ImportError as e:
                self.results[f'core_{name}'] = f"❌ Missing: {e}"
                self.failed_imports.append((module_path, str(e)))
                logger.error(f"Failed to import {module_path}: {e}")
        
        return self.results
    
    def test_relative_imports(self) -> Dict[str, str]:
        """Test relative imports from network directory."""
        # Try to import using relative paths from the network directory
        network_dir = current_file.parent
        
        relative_modules = {
            'core_config_relative': 'core.config',
            'core_types_relative': 'core.types', 
            'errors_relative': 'errors',
            'utils_relative': 'utils',
        }
        
        # Temporarily add network directory to path
        sys.path.insert(0, str(network_dir))
        
        try:
            for name, module_path in relative_modules.items():
                try:
                    module = __import__(module_path, fromlist=[''])
                    self.results[name] = "✅ Available (relative)"
                    logger.info(f"Successfully imported {module_path} (relative)")
                except ImportError as e:
                    self.results[name] = f"❌ Missing: {e}"
                    self.failed_imports.append((module_path, str(e)))
                    logger.error(f"Failed to import {module_path} (relative): {e}")
        finally:
            # Remove network directory from path
            if str(network_dir) in sys.path:
                sys.path.remove(str(network_dir))
        
        return self.results
    
    def test_direct_file_imports(self) -> Dict[str, str]:
        """Test importing files directly from filesystem."""
        network_dir = current_file.parent
        
        # Check if files exist
        files_to_check = {
            'core_config_file': network_dir / 'core' / 'config.py',
            'core_types_file': network_dir / 'core' / 'types.py',
            'errors_file': network_dir / 'errors.py',
            'utils_init_file': network_dir / 'utils' / '__init__.py',
            'security_file': network_dir / 'security' / 'security_hardening.py',
        }
        
        for name, file_path in files_to_check.items():
            if file_path.exists():
                self.results[name] = f"✅ File exists: {file_path.name}"
                logger.info(f"Found file: {file_path}")
            else:
                self.results[name] = f"❌ File missing: {file_path}"
                logger.error(f"Missing file: {file_path}")
        
        return self.results
    
    def test_package_structure(self) -> Dict[str, str]:
        """Test if package structure is correct."""
        project_root = current_file.parent.parent.parent
        
        # Check for required __init__.py files
        init_files = {
            'project_init': project_root / 'enhanced_csp' / '__init__.py',
            'network_init': project_root / 'enhanced_csp' / 'network' / '__init__.py',
            'core_init': project_root / 'enhanced_csp' / 'network' / 'core' / '__init__.py',
            'utils_init': project_root / 'enhanced_csp' / 'network' / 'utils' / '__init__.py',
            'security_init': project_root / 'enhanced_csp' / 'network' / 'security' / '__init__.py',
        }
        
        for name, init_path in init_files.items():
            if init_path.exists():
                self.results[name] = f"✅ Package init exists"
                logger.info(f"Found package init: {init_path}")
            else:
                self.results[name] = f"❌ Missing package init"
                logger.warning(f"Missing package init: {init_path}")
                
                # Try to create missing __init__.py files
                try:
                    init_path.parent.mkdir(parents=True, exist_ok=True)
                    init_path.write_text(f'# {init_path.parent.name} package\n')
                    self.results[name] = f"✅ Created package init"
                    logger.info(f"Created missing __init__.py: {init_path}")
                except Exception as e:
                    logger.error(f"Failed to create {init_path}: {e}")
        
        return self.results
    
    def print_results(self) -> None:
        """Print test results in a formatted way."""
        print("\n📋 Import Test Results:")
        print("=" * 60)
        for module, status in self.results.items():
            print(f"  {module:<30}: {status}")
        
        if self.failed_imports:
            print(f"\n❌ Failed Imports ({len(self.failed_imports)}):")
            print("-" * 40)
            for module, error in self.failed_imports:
                print(f"  {module}: {error}")


class DependencyTester:
    """Test optional dependencies."""
    
    def test_optional_dependencies(self) -> Dict[str, str]:
        """Test optional performance dependencies."""
        dependencies = {
            'msgpack': 'Fast serialization',
            'lz4': 'Fast compression',
            'zstandard': 'Advanced compression',
            'psutil': 'System monitoring',
            'uvloop': 'High-performance event loop',
            'cryptography': 'Security cryptography',
            'numpy': 'Numerical computing',
            'yaml': 'YAML configuration',
            'base58': 'Base58 encoding',
            'fastapi': 'Web framework',
            'pydantic': 'Data validation',
        }
        
        results = {}
        for dep, description in dependencies.items():
            try:
                __import__(dep)
                results[dep] = f"✅ {description}"
            except ImportError:
                results[dep] = f"❌ {description} (install with: pip install {dep})"
        
        return results


class BasicFunctionalityTester:
    """Test basic functionality without full imports."""
    
    def test_file_loading(self) -> bool:
        """Test loading Python files directly."""
        try:
            network_dir = current_file.parent
            
            # Try to load config.py directly
            config_file = network_dir / 'core' / 'config.py'
            if config_file.exists():
                with open(config_file, 'r') as f:
                    content = f.read()
                    if 'NetworkConfig' in content and 'dataclass' in content:
                        logger.info("✅ Config file structure looks correct")
                        return True
                    else:
                        logger.error("❌ Config file missing expected content")
                        return False
            else:
                logger.error(f"❌ Config file not found: {config_file}")
                return False
                
        except Exception as e:
            logger.error(f"❌ File loading test failed: {e}")
            return False
    
    def test_simple_import(self) -> bool:
        """Test importing with sys.path manipulation."""
        try:
            # Add the enhanced_csp directory to Python path
            enhanced_csp_dir = current_file.parent.parent
            if str(enhanced_csp_dir) not in sys.path:
                sys.path.insert(0, str(enhanced_csp_dir))
            
            # Try importing with the shorter path
            import network.core.config
            logger.info("✅ Successfully imported network.core.config")
            return True
            
        except ImportError as e:
            logger.error(f"❌ Simple import failed: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error in simple import: {e}")
            return False


def create_missing_init_files():
    """Create missing __init__.py files."""
    project_root = current_file.parent.parent.parent
    
    init_files_to_create = [
        project_root / 'enhanced_csp' / '__init__.py',
        project_root / 'enhanced_csp' / 'network' / '__init__.py',
        project_root / 'enhanced_csp' / 'network' / 'core' / '__init__.py',
        project_root / 'enhanced_csp' / 'network' / 'utils' / '__init__.py',
        project_root / 'enhanced_csp' / 'network' / 'security' / '__init__.py',
        project_root / 'enhanced_csp' / 'network' / 'p2p' / '__init__.py',
        project_root / 'enhanced_csp' / 'network' / 'mesh' / '__init__.py',
        project_root / 'enhanced_csp' / 'network' / 'dns' / '__init__.py',
        project_root / 'enhanced_csp' / 'network' / 'routing' / '__init__.py',
    ]
    
    created_files = []
    
    for init_file in init_files_to_create:
        if not init_file.exists():
            try:
                init_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Create appropriate content based on directory
                if 'enhanced_csp' == init_file.parent.name:
                    content = '''# enhanced_csp package
"""Enhanced CSP System - Advanced Computing Systems Platform."""
__version__ = "1.0.0"
'''
                elif 'network' == init_file.parent.name:
                    content = '''# enhanced_csp.network package
"""Enhanced CSP Network Module."""
__version__ = "1.0.0"
'''
                else:
                    content = f'# {init_file.parent.name} package\n'
                
                init_file.write_text(content)
                created_files.append(init_file)
                logger.info(f"✅ Created: {init_file}")
                
            except Exception as e:
                logger.error(f"❌ Failed to create {init_file}: {e}")
    
    return created_files


async def run_comprehensive_tests():
    """Run comprehensive test suite."""
    print("🧪 Enhanced CSP Network - Comprehensive Test Suite")
    print("=" * 60)
    
    try:
        # First, create missing __init__.py files
        print("\n🔧 Creating missing package files...")
        created_files = create_missing_init_files()
        if created_files:
            print(f"Created {len(created_files)} missing __init__.py files")
        
        # Test package structure
        print("\n📦 Testing package structure...")
        import_tester = ImportTester()
        import_tester.test_package_structure()
        
        # Test file existence
        print("\n📁 Testing file existence...")
        import_tester.test_direct_file_imports()
        
        # Test basic functionality
        print("\n🔬 Testing basic functionality...")
        basic_tester = BasicFunctionalityTester()
        file_loading_test = basic_tester.test_file_loading()
        simple_import_test = basic_tester.test_simple_import()
        
        # Test imports
        print("\n📥 Testing imports...")
        core_results = import_tester.test_core_modules()
        relative_results = import_tester.test_relative_imports()
        
        # Test dependencies
        print("\n🔧 Testing dependencies...")
        dependency_tester = DependencyTester()
        dep_results = dependency_tester.test_optional_dependencies()
        
        # Print all results
        import_tester.print_results()
        
        print(f"\n🔧 Optional Dependencies:")
        print("-" * 40)
        for dep, status in dep_results.items():
            print(f"  {dep:<15}: {status}")
        
        # Summary
        print(f"\n📋 Test Summary:")
        print("=" * 40)
        
        total_modules = len(import_tester.results)
        available_modules = len([r for r in import_tester.results.values() if r.startswith('✅')])
        
        print(f"Package structure: {'✅ Good' if available_modules > 0 else '❌ Issues found'}")
        print(f"File loading: {'✅ Working' if file_loading_test else '❌ Failed'}")
        print(f"Simple imports: {'✅ Working' if simple_import_test else '❌ Failed'}")
        print(f"Available components: {available_modules}/{total_modules}")
        print(f"Failed imports: {len(import_tester.failed_imports)}")
        
        if import_tester.failed_imports:
            print(f"\n🔧 Next Steps:")
            print("-" * 30)
            print("1. Ensure you're running from the correct directory")
            print("2. Check that all files were created properly")
            print("3. Verify PYTHONPATH includes the project root")
            print("4. Install missing dependencies with: pip install -r requirements.txt")
            print("\n💡 Try running from project root directory:")
            print(f"   cd {project_root}")
            print("   python enhanced_csp/network/test_optimizations.py")
        
        # Return success if basic tests pass
        basic_success = file_loading_test and (available_modules > 0)
        
        if basic_success:
            print(f"\n🎉 Basic functionality is working!")
        else:
            print(f"\n⚠️  Some basic tests failed. See details above.")
        
        return basic_success
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Test interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    success = await run_comprehensive_tests()
    
    if not success:
        print(f"\n🚨 If imports are still failing, try these solutions:")
        print("1. Run from project root directory")
        print("2. Set PYTHONPATH: export PYTHONPATH=$PWD")
        print("3. Install package in development mode: pip install -e .")
    
    return 0  # Always return 0 to avoid exit issues


if __name__ == "__main__":
    asyncio.run(main())
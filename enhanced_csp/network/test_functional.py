#!/usr/bin/env python3
"""
Enhanced CSP Network - Directory-Independent Functional Test
Works from any directory and properly sets up import paths.
"""

import asyncio
import sys
import time
import json
from pathlib import Path

def setup_import_paths():
    """Setup proper import paths regardless of current directory."""
    
    # Get the script location
    script_path = Path(__file__).resolve()
    
    # Find the project root by looking for enhanced_csp directory
    current_dir = script_path.parent
    project_root = None
    
    # Search up the directory tree for enhanced_csp
    for parent in [current_dir] + list(current_dir.parents):
        enhanced_csp_dir = parent / "enhanced_csp"
        if enhanced_csp_dir.exists() and enhanced_csp_dir.is_dir():
            project_root = parent
            break
    
    if project_root is None:
        print("❌ Could not find enhanced_csp directory!")
        print("Make sure you're running this from within the project structure.")
        return False
    
    # Add project root to Python path
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    
    print(f"🔍 Found project root: {project_root}")
    print(f"🔍 Current directory: {Path.cwd()}")
    print(f"🔍 Script location: {script_path}")
    print(f"🔍 Python path setup: ✅")
    
    return True

async def test_imports_basic():
    """Test basic imports to verify package structure."""
    print("📦 Testing Basic Imports...")
    
    test_imports = [
        "enhanced_csp",
        "enhanced_csp.network", 
        "enhanced_csp.network.core",
        "enhanced_csp.network.core.config",
        "enhanced_csp.network.core.types",
    ]
    
    successful_imports = []
    failed_imports = []
    
    for import_name in test_imports:
        try:
            __import__(import_name)
            successful_imports.append(import_name)
            print(f"  ✅ {import_name}")
        except ImportError as e:
            failed_imports.append((import_name, str(e)))
            print(f"  ❌ {import_name}: {e}")
    
    success_rate = len(successful_imports) / len(test_imports)
    print(f"  📊 Import success rate: {success_rate:.1%}")
    
    return success_rate > 0.5  # At least 50% of imports should work

async def test_configuration():
    """Test configuration creation and validation."""
    print("🔧 Testing Configuration System...")
    
    try:
        from enhanced_csp.network.core.config import NetworkConfig, SecurityConfig
        
        # Test basic config creation
        config = NetworkConfig()
        print(f"  ✅ Created NetworkConfig: {config.node_name}")
        
        # Test development config
        dev_config = NetworkConfig.development()
        print(f"  ✅ Created dev config: {dev_config.node_name}")
        
        # Test config serialization
        config_dict = config.to_dict()
        loaded_config = NetworkConfig.from_dict(config_dict)
        print(f"  ✅ Config serialization works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}")
        return False

async def test_types_and_messages():
    """Test core types and message creation."""
    print("🔗 Testing Types and Messages...")
    
    try:
        from enhanced_csp.network.core.types import NodeID, NetworkMessage, MessageType
        
        # Test NodeID creation
        node_id = NodeID.generate()
        print(f"  ✅ Generated NodeID: {str(node_id)[:20]}...")
        
        # Test message creation
        message = NetworkMessage.create(
            msg_type=MessageType.DATA,
            sender=node_id,
            payload={"test": "data", "timestamp": time.time()}
        )
        print(f"  ✅ Created message: {message.id[:8]}...")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Types and messages test failed: {e}")
        return False

async def test_direct_file_access():
    """Test accessing files directly without imports."""
    print("📁 Testing Direct File Access...")
    
    try:
        # Find project root
        script_path = Path(__file__).resolve()
        current_dir = script_path.parent
        
        for parent in [current_dir] + list(current_dir.parents):
            enhanced_csp_dir = parent / "enhanced_csp"
            if enhanced_csp_dir.exists():
                project_root = parent
                break
        
        # Check key files exist
        key_files = [
            "enhanced_csp/__init__.py",
            "enhanced_csp/network/__init__.py", 
            "enhanced_csp/network/core/__init__.py",
            "enhanced_csp/network/core/config.py",
            "enhanced_csp/network/core/types.py",
            "enhanced_csp/main.py",
        ]
        
        existing_files = []
        missing_files = []
        
        for file_path in key_files:
            full_path = project_root / file_path
            if full_path.exists():
                existing_files.append(file_path)
                print(f"  ✅ Found: {file_path}")
            else:
                missing_files.append(file_path)
                print(f"  ❌ Missing: {file_path}")
        
        print(f"  📊 Files found: {len(existing_files)}/{len(key_files)}")
        return len(missing_files) == 0
        
    except Exception as e:
        print(f"  ❌ File access test failed: {e}")
        return False

async def test_local_imports():
    """Test imports using local directory structure."""
    print("🔄 Testing Local Imports...")
    
    try:
        # Try to import using the current directory structure
        script_path = Path(__file__).resolve()
        current_dir = script_path.parent
        
        # Find enhanced_csp directory
        enhanced_csp_path = None
        for parent in [current_dir] + list(current_dir.parents):
            candidate = parent / "enhanced_csp"
            if candidate.exists():
                enhanced_csp_path = candidate
                break
        
        if enhanced_csp_path is None:
            print("  ❌ Could not find enhanced_csp directory")
            return False
        
        # Add the enhanced_csp parent to sys.path temporarily
        enhanced_csp_parent = str(enhanced_csp_path.parent)
        if enhanced_csp_parent not in sys.path:
            sys.path.insert(0, enhanced_csp_parent)
        
        # Now try imports
        from enhanced_csp.network.core.config import NetworkConfig
        config = NetworkConfig()
        print(f"  ✅ Successfully imported and created NetworkConfig")
        
        from enhanced_csp.network.core.types import NodeID
        node_id = NodeID.generate()
        print(f"  ✅ Successfully imported and created NodeID")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Local imports test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_environment_setup():
    """Test if the Python environment is set up correctly."""
    print("🌍 Testing Environment Setup...")
    
    try:
        # Check Python version
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        print(f"  ✅ Python version: {python_version}")
        
        # Check if we're in a virtual environment
        in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        print(f"  {'✅' if in_venv else '⚠️'} Virtual environment: {'Active' if in_venv else 'Not detected'}")
        
        # Check available packages
        required_packages = ['pathlib', 'asyncio', 'dataclasses', 'typing']
        optional_packages = ['fastapi', 'pydantic', 'cryptography', 'base58']
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"  ✅ Required package: {package}")
            except ImportError:
                print(f"  ❌ Missing required package: {package}")
        
        for package in optional_packages:
            try:
                __import__(package)
                print(f"  ✅ Optional package: {package}")
            except ImportError:
                print(f"  ⚠️ Optional package missing: {package}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Environment setup test failed: {e}")
        return False

async def run_all_tests():
    """Run all tests in order of complexity."""
    print("🧪 Enhanced CSP Network - Directory-Independent Test Suite")
    print("=" * 70)
    
    # Step 1: Setup import paths
    if not setup_import_paths():
        print("❌ Failed to setup import paths. Exiting.")
        return False
    
    print()
    
    tests = [
        ("Environment Setup", test_environment_setup),
        ("Direct File Access", test_direct_file_access),
        ("Basic Imports", test_imports_basic),
        ("Local Imports", test_local_imports),
        ("Configuration System", test_configuration),
        ("Types and Messages", test_types_and_messages),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔬 Running {test_name}...")
        try:
            success = await test_func()
            results.append((test_name, success))
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"{status} {test_name}")
        except Exception as e:
            print(f"❌ ERROR in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n📋 Test Results Summary:")
    print("=" * 40)
    
    passed_tests = [name for name, success in results if success]
    failed_tests = [name for name, success in results if not success]
    
    print(f"✅ Passed: {len(passed_tests)}/{len(results)}")
    print(f"❌ Failed: {len(failed_tests)}/{len(results)}")
    
    if passed_tests:
        print(f"\n✅ Passed Tests:")
        for test_name in passed_tests:
            print(f"   - {test_name}")
    
    if failed_tests:
        print(f"\n❌ Failed Tests:")
        for test_name in failed_tests:
            print(f"   - {test_name}")
    
    overall_success = len(failed_tests) <= 1  # Allow 1 failure
    
    if overall_success:
        print(f"\n🎉 System is mostly functional!")
        print(f"\n🚀 Try running the main application:")
        print(f"   cd to project root directory")
        print(f"   python enhanced_csp/main.py")
    else:
        print(f"\n⚠️ Multiple tests failed. Check the setup and file structure.")
        print(f"\n🔧 Troubleshooting tips:")
        print(f"   1. Make sure you're in the right directory")
        print(f"   2. Check that all __init__.py files exist")
        print(f"   3. Verify Python path includes project root")
    
    return overall_success

if __name__ == "__main__":
    asyncio.run(run_all_tests())
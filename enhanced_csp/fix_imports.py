#!/usr/bin/env python3
"""
Fix Enhanced CSP Network Package Structure
This script creates all missing __init__.py files and fixes import issues.
"""

import os
import sys
from pathlib import Path

def create_init_files():
    """Create all missing __init__.py files."""
    
    # Get current directory
    current_dir = Path.cwd()
    print(f"🔍 Working in: {current_dir}")
    
    # Define all directories that need __init__.py files
    package_dirs = [
        'enhanced_csp',
        'enhanced_csp/network',
        'enhanced_csp/network/core', 
        'enhanced_csp/network/security',
        'enhanced_csp/network/utils',
        'enhanced_csp/network/p2p',
        'enhanced_csp/network/mesh',
        'enhanced_csp/network/dns', 
        'enhanced_csp/network/routing',
        'enhanced_csp/network/examples',
        'enhanced_csp/network/tests',
    ]
    
    # Content for different __init__.py files
    init_contents = {
        'enhanced_csp': '''# enhanced_csp package
"""Enhanced CSP System - Advanced Computing Systems Platform."""
__version__ = "1.0.0"
''',
        'enhanced_csp/network': '''# enhanced_csp.network package
"""Enhanced CSP Network Module."""
__version__ = "1.0.0"

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == "NetworkConfig":
        from .core.config import NetworkConfig
        return NetworkConfig
    elif name == "NodeID":
        from .core.types import NodeID  
        return NodeID
    elif name == "MessageType":
        from .core.types import MessageType
        return MessageType
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ["NetworkConfig", "NodeID", "MessageType"]
''',
        'enhanced_csp/network/core': '''# network.core package
"""Core network components."""

try:
    from .config import NetworkConfig, SecurityConfig, P2PConfig
    from .types import NodeID, MessageType, NetworkMessage
except ImportError:
    pass

__all__ = ["NetworkConfig", "SecurityConfig", "P2PConfig", "NodeID", "MessageType", "NetworkMessage"]
''',
        'enhanced_csp/network/security': '''# network.security package  
"""Security components."""

try:
    from .security_hardening import SecurityOrchestrator, MessageValidator
except ImportError:
    pass

__all__ = ["SecurityOrchestrator", "MessageValidator"]
''',
    }
    
    created_files = []
    
    for package_dir in package_dirs:
        # Create directory if it doesn't exist
        dir_path = Path(package_dir)
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py file
        init_file = dir_path / '__init__.py'
        
        if not init_file.exists():
            # Use specific content if available, otherwise generic
            content = init_contents.get(package_dir, f'# {dir_path.name} package\n')
            
            try:
                init_file.write_text(content)
                created_files.append(init_file)
                print(f"✅ Created: {init_file}")
            except Exception as e:
                print(f"❌ Failed to create {init_file}: {e}")
        else:
            print(f"✓ Exists: {init_file}")
    
    return created_files

def check_required_files():
    """Check if required Python files exist."""
    
    required_files = [
        'enhanced_csp/network/core/config.py',
        'enhanced_csp/network/core/types.py', 
        'enhanced_csp/network/security/security_hardening.py',
        'enhanced_csp/network/errors.py',
        'enhanced_csp/network/utils/__init__.py',
        'enhanced_csp/main.py',
        'enhanced_csp/network/test_optimizations.py',
    ]
    
    missing_files = []
    existing_files = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            existing_files.append(file_path)
            print(f"✅ Found: {file_path}")
        else:
            missing_files.append(file_path)
            print(f"❌ Missing: {file_path}")
    
    return existing_files, missing_files

def fix_python_path():
    """Add current directory to Python path."""
    current_dir = str(Path.cwd())
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
        print(f"✅ Added to Python path: {current_dir}")
    else:
        print(f"✓ Already in Python path: {current_dir}")

def test_imports():
    """Test if imports work after fixes."""
    
    print("\n🧪 Testing imports...")
    
    test_imports = [
        ('enhanced_csp', 'Enhanced CSP package'),
        ('enhanced_csp.network', 'Network package'),
        ('enhanced_csp.network.core', 'Core package'),
        ('enhanced_csp.network.core.config', 'Config module'),
        ('enhanced_csp.network.core.types', 'Types module'),
        ('enhanced_csp.network.errors', 'Errors module'),
        ('enhanced_csp.network.utils', 'Utils package'),
    ]
    
    successful_imports = []
    failed_imports = []
    
    for import_path, description in test_imports:
        try:
            __import__(import_path)
            successful_imports.append((import_path, description))
            print(f"✅ {description}: {import_path}")
        except ImportError as e:
            failed_imports.append((import_path, description, str(e)))
            print(f"❌ {description}: {import_path} - {e}")
    
    return successful_imports, failed_imports

def create_requirements_txt():
    """Create a requirements.txt file if it doesn't exist."""
    
    requirements_file = Path('requirements.txt')
    
    if not requirements_file.exists():
        requirements_content = """# Enhanced CSP Network Requirements

# Core dependencies
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0
pathlib
dataclasses; python_version<"3.7"

# Security
cryptography>=41.0.0

# Encoding
base58>=2.1.0

# Optional performance dependencies
msgpack>=1.0.0
lz4>=4.0.0
zstandard>=0.21.0
psutil>=5.9.0
uvloop>=0.17.0

# Optional AI dependencies
# torch>=2.0.0
# transformers>=4.30.0
# numpy>=1.24.0

# Development dependencies
# pytest>=7.0.0
# pytest-asyncio>=0.21.0
# black>=23.0.0
# isort>=5.12.0
# mypy>=1.5.0
"""
        
        try:
            requirements_file.write_text(requirements_content)
            print(f"✅ Created: {requirements_file}")
        except Exception as e:
            print(f"❌ Failed to create requirements.txt: {e}")
    else:
        print(f"✓ Exists: {requirements_file}")

def main():
    """Main function to fix all import issues."""
    
    print("🔧 Enhanced CSP Network - Package Structure Fixer")
    print("=" * 60)
    
    # Step 1: Fix Python path
    print("\n1️⃣ Fixing Python path...")
    fix_python_path()
    
    # Step 2: Check required files
    print("\n2️⃣ Checking required files...")
    existing_files, missing_files = check_required_files()
    
    # Step 3: Create __init__.py files
    print("\n3️⃣ Creating package structure...")
    created_files = create_init_files()
    
    # Step 4: Create requirements.txt
    print("\n4️⃣ Checking requirements.txt...")
    create_requirements_txt()
    
    # Step 5: Test imports
    print("\n5️⃣ Testing imports...")
    successful_imports, failed_imports = test_imports()
    
    # Summary
    print("\n📋 Summary:")
    print("=" * 30)
    print(f"✅ Existing files: {len(existing_files)}")
    print(f"❌ Missing files: {len(missing_files)}")
    print(f"📁 Created __init__.py files: {len(created_files)}")
    print(f"✅ Successful imports: {len(successful_imports)}")
    print(f"❌ Failed imports: {len(failed_imports)}")
    
    if missing_files:
        print(f"\n⚠️ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
    
    if failed_imports:
        print(f"\n⚠️ Failed imports:")
        for import_path, description, error in failed_imports:
            print(f"   - {import_path}: {error}")
    
    # Next steps
    print(f"\n🚀 Next steps:")
    if len(successful_imports) > len(failed_imports):
        print("1. Package structure is mostly working!")
        print("2. Run tests: python enhanced_csp/network/test_optimizations.py")
        print("3. Start system: python enhanced_csp/main.py")
    else:
        print("1. Ensure you have all the required Python files")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Run this script again to verify fixes")
    
    if len(missing_files) == 0 and len(failed_imports) <= 2:
        print("\n🎉 Package structure successfully fixed!")
        return True
    else:
        print("\n⚠️ Some issues remain. See details above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
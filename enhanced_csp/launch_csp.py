#!/usr/bin/env python3
"""Working production launcher for Enhanced CSP System."""

import os
import sys
import logging
from pathlib import Path

def setup_python_path():
    """Setup Python path correctly."""
    # Get current directory (should be the project root)
    current_dir = Path.cwd()
    
    # Add current directory to Python path  
    current_dir_str = str(current_dir)
    if current_dir_str not in sys.path:
        sys.path.insert(0, current_dir_str)
    
    print(f"🔍 Working directory: {current_dir}")
    print(f"🔍 Python path setup: ✅")
    
    # Check if enhanced_csp directory exists
    enhanced_csp_dir = current_dir / "enhanced_csp"
    if enhanced_csp_dir.exists():
        print(f"✅ Found enhanced_csp directory")
        return True
    else:
        print(f"❌ enhanced_csp directory not found at {enhanced_csp_dir}")
        return False

def test_imports():
    """Test if we can import our modules."""
    print("\n🧪 Testing imports...")
    
    try:
        # Test enhanced_csp package
        import enhanced_csp
        print("✅ enhanced_csp package imported")
        
        # Test network package
        from enhanced_csp import network
        print("✅ enhanced_csp.network imported") 
        
        # Test core config
        from enhanced_csp.network.core.config import NetworkConfig
        config = NetworkConfig()
        print(f"✅ NetworkConfig created: {config.node_name}")
        
        # Test main app
        from enhanced_csp.main import app
        print("✅ FastAPI app imported")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def main():
    """Launch the Enhanced CSP System."""
    
    print("🚀 Enhanced CSP System - Working Production Launcher")
    print("=" * 60)
    
    # Setup environment to avoid file watching issues
    os.environ['WATCHFILES_FORCE_POLLING'] = 'true'
    
    # Setup Python path
    if not setup_python_path():
        print("❌ Could not find enhanced_csp directory")
        print("Make sure you're running this from the project root directory")
        sys.exit(1)
    
    # Test imports
    if not test_imports():
        print("❌ Import test failed")
        print("Check that all files are in place")
        sys.exit(1)
    
    try:
        print("\n🌐 Starting Enhanced CSP System...")
        
        import uvicorn
        from enhanced_csp.main import app
        
        print("✅ All imports successful")
        print("🚀 Starting server in production mode...")
        print()
        print("📍 Dashboard: http://localhost:8000/dashboard")
        print("📚 API Docs: http://localhost:8000/docs")
        print("💚 Health: http://localhost:8000/health")
        print("🔬 Test CSP: http://localhost:8000/api/test/run-diagnostics")
        print()
        print("Press Ctrl+C to stop the server")
        print("=" * 60)
        
        # Run in production mode (no file watching)
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            reload=False,           # Disable reload/file watching
            access_log=True,
            log_level="info",
            workers=1,
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're in the project root directory")
        print("2. Check that enhanced_csp/__init__.py exists")
        print("3. Check that enhanced_csp/network/__init__.py exists")
        print("4. Try: ls enhanced_csp/")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        print("✅ Enhanced CSP System shutdown complete")
    except Exception as e:
        print(f"❌ Server error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
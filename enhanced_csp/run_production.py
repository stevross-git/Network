#!/usr/bin/env python3
"""Production launcher for Enhanced CSP System."""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set environment variable to disable file watching
os.environ['WATCHFILES_FORCE_POLLING'] = 'true'

def main():
    """Launch the Enhanced CSP System in production mode."""
    
    print("🚀 Enhanced CSP System - Production Launcher")
    print("=" * 50)
    
    try:
        import uvicorn
        from enhanced_csp.main import app
        
        print("✅ Successfully imported application")
        print("🌐 Starting server...")
        print("📍 Dashboard: http://localhost:8000/dashboard")
        print("📚 API Docs: http://localhost:8000/docs")
        print("💔 Health: http://localhost:8000/health")
        print()
        print("Press Ctrl+C to stop the server")
        print("=" * 50)
        
        # Run in production mode (no reload, no file watching)
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            reload=False,           # Disable reload to avoid file watching
            access_log=True,        # Enable access logs
            log_level="info",       # Set log level
            workers=1,              # Single worker for development
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Install dependencies: pip install fastapi uvicorn")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
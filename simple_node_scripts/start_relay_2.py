#!/usr/bin/env python3
"""
Simple startup script for node: relay_2
Uses network_startup.py directly with command line arguments
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Start the relay_2 node using network_startup.py"""
    
    print(f"🚀 Starting relay_2 node...")
    print(f"📡 Port: 30402")
    print(f"🏷️  Type: relay")
    print(f"🌐 Genesis: genesis.peoplesainetwork.com:30300")
    
    # Check if network_startup.py exists
    startup_script = Path("network_startup.py")
    if not startup_script.exists():
        print("❌ network_startup.py not found in current directory")
        return 1
    
    # Build command line arguments for network_startup.py
    cmd = [
        sys.executable, "network_startup.py",
        "--genesis-host", "genesis.peoplesainetwork.com",
        "--genesis-port", "30300",
        "--local-port", "30402",
        "--node-name", "relay_2",
        "--data-dir", "./node_data/relay_2"
    ]
    
    # Add node type specific flags if available
    if "relay" == "super_peer":
        # Super peer might have additional capabilities
        pass
    elif "relay" == "edge":
        cmd.extend(["--disable-features"])  # Edge nodes with minimal features
    
    print(f"🔧 Executing: {' '.join(cmd)}")
    
    try:
        # Run the network startup script
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\n🛑 Node shutdown requested")
        return 0
    except Exception as e:
        print(f"❌ Error starting node: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

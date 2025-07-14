#!/usr/bin/env python3
"""
Simple Node Starter - Works directly with network_startup.py
This script creates and starts nodes using the existing network_startup.py
"""

import asyncio
import subprocess
import sys
import time
import json
from pathlib import Path
from typing import List, Dict

def create_simple_node_script(node_name: str, port: int, node_type: str = "relay", 
                             genesis_host: str = "genesis.peoplesainetwork.com",
                             genesis_port: int = 30300) -> Path:
    """Create a simple node startup script that uses network_startup.py directly"""
    
    scripts_dir = Path("./simple_node_scripts")
    scripts_dir.mkdir(exist_ok=True)
    
    script_file = scripts_dir / f"start_{node_name}.py"
    
    # Create data directory for this node
    data_dir = Path(f"./node_data/{node_name}")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    script_content = f'''#!/usr/bin/env python3
"""
Simple startup script for node: {node_name}
Uses network_startup.py directly with command line arguments
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Start the {node_name} node using network_startup.py"""
    
    print(f"🚀 Starting {node_name} node...")
    print(f"📡 Port: {port}")
    print(f"🏷️  Type: {node_type}")
    print(f"🌐 Genesis: {genesis_host}:{genesis_port}")
    
    # Check if network_startup.py exists
    startup_script = Path("network_startup.py")
    if not startup_script.exists():
        print("❌ network_startup.py not found in current directory")
        return 1
    
    # Build command line arguments for network_startup.py
    cmd = [
        sys.executable, "network_startup.py",
        "--genesis-host", "{genesis_host}",
        "--genesis-port", "{genesis_port}",
        "--local-port", "{port}",
        "--node-name", "{node_name}",
        "--data-dir", "./node_data/{node_name}"
    ]
    
    # Add node type specific flags if available
    if "{node_type}" == "super_peer":
        # Super peer might have additional capabilities
        pass
    elif "{node_type}" == "edge":
        cmd.extend(["--disable-features"])  # Edge nodes with minimal features
    
    print(f"🔧 Executing: {{' '.join(cmd)}}")
    
    try:
        # Run the network startup script
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\\n🛑 Node shutdown requested")
        return 0
    except Exception as e:
        print(f"❌ Error starting node: {{e}}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
'''
    
    with open(script_file, 'w') as f:
        f.write(script_content)
    
    # Make script executable
    script_file.chmod(0o755)
    
    print(f"📜 Created simple startup script: {script_file}")
    return script_file


def create_batch_starter(node_configs: List[Dict]) -> Path:
    """Create a batch script to start multiple nodes"""
    
    batch_file = Path("./start_all_nodes.py")
    
    batch_content = '''#!/usr/bin/env python3
"""
Batch Node Starter - Start all nodes with proper delays
"""

import subprocess
import sys
import time
from pathlib import Path

# Node configurations
NODES = [
'''
    
    for config in node_configs:
        batch_content += f'    {{"name": "{config["name"]}", "script": "{config["script"]}", "delay": {config.get("delay", 2)}}},\n'
    
    batch_content += ''']

def main():
    """Start all nodes with delays"""
    print("🚀 Starting all Enhanced CSP Network nodes...")
    print(f"📊 Total nodes: {len(NODES)}")
    print()
    
    processes = []
    
    try:
        for i, node in enumerate(NODES):
            print(f"🔧 Starting node {i+1}/{len(NODES)}: {node['name']}")
            
            # Start the node script
            process = subprocess.Popen([
                sys.executable, node['script']
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            processes.append({
                'name': node['name'],
                'process': process,
                'script': node['script']
            })
            
            print(f"✅ Started {node['name']} (PID: {process.pid})")
            
            # Wait before starting next node
            if i < len(NODES) - 1:  # Don't wait after the last node
                print(f"⏳ Waiting {node['delay']} seconds before next node...")
                time.sleep(node['delay'])
        
        print()
        print("🎉 All nodes started successfully!")
        print("📋 Node Status:")
        
        for proc_info in processes:
            if proc_info['process'].poll() is None:
                status = "RUNNING"
            else:
                status = "FAILED"
            print(f"  {proc_info['name']}: {status} (PID: {proc_info['process'].pid})")
        
        print()
        print("💡 Tips:")
        print("  - Check individual node logs for detailed output")
        print("  - Use Ctrl+C to stop this script (nodes will continue running)")
        print("  - Kill individual processes to stop specific nodes")
        
        # Keep script running to monitor
        try:
            while True:
                time.sleep(10)
                # Check if any processes have died
                for proc_info in processes:
                    if proc_info['process'].poll() is not None:
                        print(f"⚠️  Node {proc_info['name']} has stopped")
        except KeyboardInterrupt:
            print("\\n🛑 Batch starter stopping (nodes continue running)")
            
    except Exception as e:
        print(f"❌ Error in batch starter: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''
    
    with open(batch_file, 'w') as f:
        f.write(batch_content)
    
    batch_file.chmod(0o755)
    print(f"📦 Created batch starter: {batch_file}")
    return batch_file


def main():
    """Main function to create simple nodes"""
    print("====================================")
    print("Enhanced CSP Network - Simple Nodes")
    print("====================================")
    print()
    
    # Define simple node configurations
    node_configs = [
        {"name": "relay_1", "port": 30401, "type": "relay", "delay": 2},
        {"name": "relay_2", "port": 30402, "type": "relay", "delay": 2}, 
        {"name": "storage_1", "port": 30403, "type": "storage", "delay": 3},
        {"name": "compute_1", "port": 30404, "type": "compute", "delay": 2},
    ]
    
    genesis_host = "genesis.peoplesainetwork.com"
    genesis_port = 30300
    
    print(f"🔧 Creating {len(node_configs)} simple nodes...")
    print(f"🌐 Genesis server: {genesis_host}:{genesis_port}")
    print()
    
    created_scripts = []
    
    # Create individual node scripts
    for config in node_configs:
        script_file = create_simple_node_script(
            node_name=config["name"],
            port=config["port"], 
            node_type=config["type"],
            genesis_host=genesis_host,
            genesis_port=genesis_port
        )
        
        config["script"] = str(script_file)
        created_scripts.append(config)
    
    # Create batch starter
    batch_file = create_batch_starter(created_scripts)
    
    print()
    print("✅ Simple node creation complete!")
    print()
    print("🚀 To start all nodes:")
    print(f"   python3 {batch_file}")
    print()
    print("🔧 To start individual nodes:")
    for config in created_scripts:
        print(f"   python3 {config['script']}  # {config['name']}")
    print()
    print("📁 Node data will be stored in: ./node_data/")
    print("📜 Scripts are stored in: ./simple_node_scripts/")


if __name__ == "__main__":
    main()

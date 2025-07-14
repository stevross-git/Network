#!/usr/bin/env python3
"""
Batch Node Starter - Start all nodes with proper delays
"""

import subprocess
import sys
import time
from pathlib import Path

# Node configurations
NODES = [
    {"name": "relay_1", "script": "simple_node_scripts/start_relay_1.py", "delay": 2},
    {"name": "relay_2", "script": "simple_node_scripts/start_relay_2.py", "delay": 2},
    {"name": "storage_1", "script": "simple_node_scripts/start_storage_1.py", "delay": 3},
    {"name": "compute_1", "script": "simple_node_scripts/start_compute_1.py", "delay": 2},
]

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
            print("\n🛑 Batch starter stopping (nodes continue running)")
            
    except Exception as e:
        print(f"❌ Error in batch starter: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

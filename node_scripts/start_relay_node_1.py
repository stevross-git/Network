#!/usr/bin/env python3
"""
Startup script for Enhanced CSP Network node: relay_node_1
Generated automatically by node_creator.py
"""

import asyncio
import sys
import json
from pathlib import Path

# Add the enhanced_csp directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from enhanced_csp.network.network_startup import GenesisConnector
    from enhanced_csp.network.core.config import NetworkConfig
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Enhanced CSP imports not available: {e}")
    IMPORTS_AVAILABLE = False

async def main():
    """Start the relay_node_1 node"""
    if not IMPORTS_AVAILABLE:
        print("❌ Required modules not available")
        return
    
    # Load configuration
    config_file = Path("node_configs/relay_node_1_config.json")
    with open(config_file, 'r') as f:
        config_data = json.load(f)
    
    print(f"🚀 Starting node: relay_node_1")
    print(f"📋 Config: {config_file}")
    print(f"🌐 Type: {config_data['network']['node_type']}")
    print(f"📡 Port: {config_data['p2p']['listen_port']}")
    
    # Create connector
    connector = GenesisConnector()
    connector.setup_logging("INFO")
    
    try:
        # Create network configuration
        config = NetworkConfig()
        
        # Apply loaded configuration
        config.node_name = config_data['network']['node_name']
        config.node_type = config_data['network']['node_type']
        config.data_dir = Path(config_data['network']['data_dir'])
        config.data_dir.mkdir(parents=True, exist_ok=True)
        
        # P2P configuration
        config.p2p.listen_port = config_data['p2p']['listen_port']
        config.p2p.max_connections = config_data['p2p']['max_connections']
        config.p2p.bootstrap_nodes = config_data['p2p']['bootstrap_nodes']
        
        # Capabilities
        config.capabilities.relay = config_data['capabilities']['relay']
        config.capabilities.storage = config_data['capabilities']['storage']
        config.capabilities.compute = config_data['capabilities']['compute']
        config.capabilities.quantum = config_data['capabilities']['quantum']
        
        # Start network
        if await connector.start_network(config):
            print("✅ Network started successfully!")
            
            # Connect to genesis
            if await connector.connect_to_genesis():
                print("🌟 Connected to genesis network!")
            
            # Run network loop
            await connector.run_network_loop()
        else:
            print("❌ Failed to start network")
            
    except KeyboardInterrupt:
        print("🛑 Shutting down...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await connector.shutdown()
        print("👋 Node shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())

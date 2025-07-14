#!/usr/bin/env python3
"""
Enhanced CSP Network - Node Creation and Management Scripts
Create and manage multiple network nodes with different configurations
"""

import asyncio
import argparse
import time
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)8s | %(name)s | %(message)s'
)
logger = logging.getLogger("node_creator")

# Configuration templates for different node types
NODE_TEMPLATES = {
    "relay": {
        "node_type": "relay",
        "capabilities": {
            "relay": True,
            "storage": False,
            "compute": False,
            "quantum": False
        },
        "p2p": {
            "max_connections": 100,
            "enable_upnp": True,
            "enable_nat_traversal": True
        },
        "mesh": {
            "topology_type": "partial_mesh",
            "max_peers": 50
        }
    },
    
    "storage": {
        "node_type": "storage",
        "capabilities": {
            "relay": True,
            "storage": True,
            "compute": False,
            "quantum": False
        },
        "p2p": {
            "max_connections": 75,
            "enable_upnp": True
        },
        "mesh": {
            "topology_type": "hierarchical",
            "max_peers": 30
        }
    },
    
    "compute": {
        "node_type": "compute",
        "capabilities": {
            "relay": True,
            "storage": False,
            "compute": True,
            "quantum": False
        },
        "p2p": {
            "max_connections": 50,
            "enable_upnp": True
        },
        "mesh": {
            "topology_type": "dynamic_partial",
            "max_peers": 25
        }
    },
    
    "super_peer": {
        "node_type": "super_peer",
        "capabilities": {
            "relay": True,
            "storage": True,
            "compute": True,
            "quantum": False
        },
        "p2p": {
            "max_connections": 200,
            "enable_upnp": True,
            "enable_nat_traversal": True
        },
        "mesh": {
            "topology_type": "hierarchical",
            "max_peers": 100,
            "is_super_peer": True
        }
    },
    
    "edge": {
        "node_type": "edge",
        "capabilities": {
            "relay": False,
            "storage": False,
            "compute": False,
            "quantum": False
        },
        "p2p": {
            "max_connections": 20,
            "enable_upnp": True
        },
        "mesh": {
            "topology_type": "partial_mesh",
            "max_peers": 10
        }
    }
}


class NodeCreator:
    """Creates and manages multiple Enhanced CSP Network nodes"""
    
    def __init__(self, genesis_host: str = "genesis.peoplesainetwork.com", 
                 genesis_port: int = 30300):
        self.genesis_host = genesis_host
        self.genesis_port = genesis_port
        self.nodes: Dict[str, Any] = {}
        self.base_port = 30400  # Starting port for nodes
        self.node_processes = {}
        
    def create_node_config(self, node_name: str, node_type: str = "relay", 
                          port_offset: int = 0, custom_config: Dict = None) -> Dict[str, Any]:
        """Create configuration for a specific node"""
        
        # Get base template
        if node_type not in NODE_TEMPLATES:
            logger.warning(f"Unknown node type {node_type}, using 'relay'")
            node_type = "relay"
        
        config = NODE_TEMPLATES[node_type].copy()
        
        # Apply custom configuration
        if custom_config:
            self._deep_update(config, custom_config)
        
        # Set node-specific values
        local_port = self.base_port + port_offset
        
        full_config = {
            "network": {
                "node_name": node_name,
                "node_type": config["node_type"],
                "network_id": "enhanced-csp-mainnet",
                "protocol_version": "1.0.0",
                "data_dir": f"./network_data/{node_name}"
            },
            
            "p2p": {
                "listen_address": "0.0.0.0",
                "listen_port": local_port,
                "enable_mdns": True,
                "enable_upnp": config["p2p"]["enable_upnp"],
                "enable_nat_traversal": config["p2p"].get("enable_nat_traversal", True),
                "connection_timeout": 30,
                "max_connections": config["p2p"]["max_connections"],
                "bootstrap_nodes": [
                    f"/ip4/{self.genesis_host}/tcp/{self.genesis_port}",
                    "/ip4/147.75.77.187/tcp/30300",  # Backup genesis
                    "/dns4/seed1.peoplesainetwork.com/tcp/30300",
                    "/dns4/seed2.peoplesainetwork.com/tcp/30300"
                ],
                "dns_seed_domain": "peoplesainetwork.com",
                "bootstrap_retry_delay": 5,
                "max_bootstrap_attempts": 10
            },
            
            "mesh": {
                "enable_super_peers": True,
                "max_peers": config["mesh"]["max_peers"],
                "topology_type": config["mesh"]["topology_type"],
                "heartbeat_interval": 30,
                "redundancy_factor": 3,
                "is_super_peer": config["mesh"].get("is_super_peer", False)
            },
            
            "security": {
                "enable_encryption": True,
                "enable_authentication": True,
                "key_size": 2048,
                "enable_ca_mode": False,
                "trust_anchors": []
            },
            
            "capabilities": config["capabilities"],
            
            "features": {
                "enable_dht": True,
                "enable_mesh": True,
                "enable_dns": True,
                "enable_adaptive_routing": True,
                "enable_metrics": True,
                "enable_ipv6": True,
                "enable_compression": True
            },
            
            "performance": {
                "max_message_size": 1048576,  # 1MB
                "gossip_interval": 5,
                "gossip_fanout": 6,
                "metrics_interval": 60
            }
        }
        
        return full_config
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> None:
        """Deep update dictionary"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def save_node_config(self, node_name: str, config: Dict[str, Any]) -> Path:
        """Save node configuration to file"""
        config_dir = Path("./node_configs")
        config_dir.mkdir(exist_ok=True)
        
        config_file = config_dir / f"{node_name}_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"💾 Saved configuration for {node_name} to {config_file}")
        return config_file
    
    def create_startup_script(self, node_name: str, config_file: Path) -> Path:
        """Create startup script for a node"""
        scripts_dir = Path("./node_scripts")
        scripts_dir.mkdir(exist_ok=True)
        
        script_file = scripts_dir / f"start_{node_name}.py"
        
        script_content = f'''#!/usr/bin/env python3
"""
Startup script for Enhanced CSP Network node: {node_name}
Generated automatically by node_creator.py
"""

import asyncio
import sys
import json
from pathlib import Path

# Add the enhanced_csp directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    # Import from the root directory where network_startup.py is located
    from network_startup import GenesisConnector
    # Try to import NetworkConfig from enhanced_csp if available
    try:
        from enhanced_csp.network.core.config import NetworkConfig
    except ImportError:
        # Fallback - we'll create a basic config class
        NetworkConfig = None
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Enhanced CSP imports not available: {{e}}")
    IMPORTS_AVAILABLE = False

async def main():
    """Start the {node_name} node"""
    if not IMPORTS_AVAILABLE:
        print("❌ Required modules not available")
        return
    
    # Load configuration
    config_file = Path("{config_file}")
    with open(config_file, 'r') as f:
        config_data = json.load(f)
    
    print(f"🚀 Starting node: {node_name}")
    print(f"📋 Config: {{config_file}}")
    print(f"🌐 Type: {{config_data['network']['node_type']}}")
    print(f"📡 Port: {{config_data['p2p']['listen_port']}}")
    
    # Create connector
    connector = GenesisConnector()
    connector.setup_logging("INFO")
    
    try:
        # Create network configuration
        if NetworkConfig:
            config = NetworkConfig()
        else:
            # Create a simple config object if NetworkConfig not available
            class SimpleConfig:
                def __init__(self):
                    self.node_name = config_data['network']['node_name']
                    self.node_type = config_data['network']['node_type']
                    self.data_dir = Path(config_data['network']['data_dir'])
                    
                    # Create a simple P2P config
                    class P2PConfig:
                        def __init__(self):
                            self.listen_port = config_data['p2p']['listen_port']
                            self.max_connections = config_data['p2p']['max_connections']
                            self.bootstrap_nodes = config_data['p2p']['bootstrap_nodes']
                    
                    # Create a simple capabilities config
                    class Capabilities:
                        def __init__(self):
                            self.relay = config_data['capabilities']['relay']
                            self.storage = config_data['capabilities']['storage']
                            self.compute = config_data['capabilities']['compute']
                            self.quantum = config_data['capabilities']['quantum']
                    
                    self.p2p = P2PConfig()
                    self.capabilities = Capabilities()
            
            config = SimpleConfig()
        
        # Ensure data directory exists
        config.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Apply loaded configuration if using full NetworkConfig
        if NetworkConfig:
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
        print(f"❌ Error: {{e}}")
    finally:
        await connector.shutdown()
        print("👋 Node shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())
'''
        
        with open(script_file, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        script_file.chmod(0o755)
        
        logger.info(f"📜 Created startup script for {node_name}: {script_file}")
        return script_file
    
    def create_multiple_nodes(self, node_specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create multiple nodes from specifications"""
        created_nodes = []
        
        for i, spec in enumerate(node_specs):
            node_name = spec.get("name", f"node_{i}")
            node_type = spec.get("type", "relay")
            custom_config = spec.get("config", {})
            
            logger.info(f"🔧 Creating node: {node_name} (type: {node_type})")
            
            # Create configuration
            config = self.create_node_config(
                node_name=node_name,
                node_type=node_type,
                port_offset=i,
                custom_config=custom_config
            )
            
            # Save configuration
            config_file = self.save_node_config(node_name, config)
            
            # Create startup script
            script_file = self.create_startup_script(node_name, config_file)
            
            node_info = {
                "name": node_name,
                "type": node_type,
                "port": config["p2p"]["listen_port"],
                "config_file": str(config_file),
                "script_file": str(script_file),
                "config": config
            }
            
            created_nodes.append(node_info)
            self.nodes[node_name] = node_info
            
            logger.info(f"✅ Node {node_name} created successfully")
        
        return created_nodes
    
    def create_network_overview(self, nodes: List[Dict[str, Any]]) -> None:
        """Create an overview of the created network"""
        overview_file = Path("./NETWORK_OVERVIEW.md")
        
        overview_content = f"""# Enhanced CSP Network Overview

Created: {time.strftime('%Y-%m-%d %H:%M:%S')}
Genesis Server: {self.genesis_host}:{self.genesis_port}
Total Nodes: {len(nodes)}

## Node Configuration

| Node Name | Type | Port | Capabilities | Script |
|-----------|------|------|--------------|--------|
"""
        
        for node in nodes:
            caps = node["config"]["capabilities"]
            cap_str = ", ".join([k for k, v in caps.items() if v])
            overview_content += f"| {node['name']} | {node['type']} | {node['port']} | {cap_str} | `{node['script_file']}` |\n"
        
        overview_content += f"""
## Starting the Network

### Start All Nodes
```bash
# Start each node in a separate terminal
"""
        
        for node in nodes:
            overview_content += f"python3 {node['script_file']}  # {node['name']}\n"
        
        overview_content += f"""```

### Monitor Network
```bash
# Check network status
python3 enhanced_csp/network/monitoring.py

# View network dashboard
python3 dashboard_server.py
```

## Node Types

- **Relay Nodes**: Forward messages, basic networking
- **Storage Nodes**: Provide distributed storage
- **Compute Nodes**: Execute distributed computations  
- **Super Peers**: High-capacity nodes for network stability
- **Edge Nodes**: Lightweight nodes for IoT/mobile devices

## Network Topology

The network uses adaptive mesh topology with:
- Bootstrap connections to genesis server
- Peer discovery via mDNS and DHT
- Automatic topology optimization
- BATMAN routing protocol for efficient message delivery

## Configuration Files

Node configurations are stored in `./node_configs/`
Startup scripts are stored in `./node_scripts/`
"""
        
        with open(overview_file, 'w') as f:
            f.write(overview_content)
        
        logger.info(f"📋 Created network overview: {overview_file}")


def main():
    """Main function for node creation CLI"""
    parser = argparse.ArgumentParser(description="Create Enhanced CSP Network nodes")
    parser.add_argument("--genesis-host", default="genesis.peoplesainetwork.com",
                       help="Genesis server hostname")
    parser.add_argument("--genesis-port", type=int, default=30300,
                       help="Genesis server port")
    parser.add_argument("--preset", choices=["development", "production", "testing"],
                       default="development", help="Node configuration preset")
    parser.add_argument("--custom-spec", help="JSON file with custom node specifications")
    
    args = parser.parse_args()
    
    # Initialize node creator
    creator = NodeCreator(args.genesis_host, args.genesis_port)
    
    # Define node specifications based on preset
    if args.preset == "development":
        node_specs = [
            {"name": "relay_node_1", "type": "relay"},
            {"name": "relay_node_2", "type": "relay"},
            {"name": "storage_node_1", "type": "storage"},
            {"name": "compute_node_1", "type": "compute"}
        ]
    elif args.preset == "production":
        node_specs = [
            {"name": "super_peer_1", "type": "super_peer"},
            {"name": "super_peer_2", "type": "super_peer"},
            {"name": "relay_node_1", "type": "relay"},
            {"name": "relay_node_2", "type": "relay"},
            {"name": "relay_node_3", "type": "relay"},
            {"name": "storage_node_1", "type": "storage"},
            {"name": "storage_node_2", "type": "storage"},
            {"name": "compute_node_1", "type": "compute"},
            {"name": "compute_node_2", "type": "compute"}
        ]
    elif args.preset == "testing":
        node_specs = [
            {"name": "test_relay", "type": "relay"},
            {"name": "test_edge", "type": "edge"}
        ]
    
    # Load custom specifications if provided
    if args.custom_spec:
        with open(args.custom_spec, 'r') as f:
            node_specs = json.load(f)
    
    logger.info(f"🚀 Creating {len(node_specs)} nodes with {args.preset} preset")
    
    # Create nodes
    created_nodes = creator.create_multiple_nodes(node_specs)
    
    # Create network overview
    creator.create_network_overview(created_nodes)
    
    logger.info("🎉 Node creation complete!")
    logger.info("📋 Check NETWORK_OVERVIEW.md for startup instructions")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Network Configuration Helper
Ensures proper configuration for connecting to the genesis server.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


class NetworkConfigHelper:
    """Helper class to create and validate network configurations."""
    
    def __init__(self):
        self.config_dir = Path("./config")
        self.config_dir.mkdir(exist_ok=True)
        
    def create_genesis_connection_config(self, 
                                       genesis_host: str = "genesis.peoplesainetwork.com",
                                       genesis_port: int = 30300,
                                       local_port: int = 30301,
                                       node_name: str = "csp-node") -> Dict[str, Any]:
        """Create a complete configuration for genesis connection."""
        
        config = {
            "network": {
                "node_name": node_name,
                "node_type": "standard",
                "network_id": "enhanced-csp-mainnet",
                "protocol_version": "1.0.0",
                "data_dir": "./network_data"
            },
            
            "p2p": {
                "listen_address": "0.0.0.0",
                "listen_port": local_port,
                "enable_mdns": True,
                "enable_upnp": True,
                "enable_nat_traversal": True,
                "connection_timeout": 30,
                "max_connections": 100,
                "bootstrap_nodes": [
                    f"/ip4/{genesis_host}/tcp/{genesis_port}",
                    "/ip4/147.75.77.187/tcp/30300",  # Backup genesis
                    "/dns4/seed1.peoplesainetwork.com/tcp/30300",
                    "/dns4/seed2.peoplesainetwork.com/tcp/30300",
                    "/dns4/bootstrap.peoplesainetwork.com/tcp/30300"
                ],
                "dns_seed_domain": "peoplesainetwork.com",
                "bootstrap_retry_delay": 5,
                "max_bootstrap_attempts": 10
            },
            
            "mesh": {
                "enable_super_peers": True,
                "max_peers": 50,
                "topology_type": "adaptive_partial",
                "heartbeat_interval": 30,
                "redundancy_factor": 3
            },
            
            "security": {
                "enable_encryption": True,
                "enable_authentication": True,
                "key_size": 2048,
                "enable_ca_mode": False,
                "trust_anchors": []
            },
            
            "capabilities": {
                "relay": True,
                "storage": False,
                "compute": False,
                "quantum": False
            },
            
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
        
        return config
    
    def save_config(self, config: Dict[str, Any], filename: str = "network_config.json"):
        """Save configuration to file."""
        config_path = self.config_dir / filename
        
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"✅ Configuration saved to {config_path}")
            return str(config_path)
        except Exception as e:
            logger.error(f"❌ Failed to save config: {e}")
            return None
    
    def load_config(self, filename: str = "network_config.json") -> Dict[str, Any]:
        """Load configuration from file."""
        config_path = self.config_dir / filename
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"✅ Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"⚠️  Config file not found: {config_path}")
            return {}
        except Exception as e:
            logger.error(f"❌ Failed to load config: {e}")
            return {}
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate configuration and return any issues."""
        issues = []
        
        # Check required sections
        required_sections = ["network", "p2p", "mesh", "security", "capabilities", "features"]
        for section in required_sections:
            if section not in config:
                issues.append(f"Missing required section: {section}")
        
        # Validate P2P configuration
        if "p2p" in config:
            p2p = config["p2p"]
            if not p2p.get("bootstrap_nodes"):
                issues.append("No bootstrap nodes configured")
            
            if not isinstance(p2p.get("listen_port"), int):
                issues.append("Invalid listen_port")
                
            if p2p.get("listen_port", 0) < 1024 or p2p.get("listen_port", 0) > 65535:
                issues.append("listen_port should be between 1024-65535")
        
        # Validate network section
        if "network" in config:
            network = config["network"]
            if not network.get("node_name"):
                issues.append("node_name is required")
                
            if not network.get("network_id"):
                issues.append("network_id is required")
        
        return issues
    
    def create_startup_script(self, config_file: str) -> str:
        """Create a startup script for the network."""
        
        script_content = f"""#!/bin/bash
# Enhanced CSP Network Startup Script
# Generated automatically by NetworkConfigHelper

echo "🚀 Starting Enhanced CSP Network..."
echo "📋 Using config: {config_file}"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not found"
    exit 1
fi

# Check if required files exist
if [ ! -f "{config_file}" ]; then
    echo "❌ Configuration file not found: {config_file}"
    exit 1
fi

# Set environment variables
export PYTHONPATH="$PWD:$PYTHONPATH"
export CSP_NETWORK_CONFIG="{config_file}"

# Start the network
echo "🔧 Initializing network with configuration..."
python3 enhanced_csp/run_network.py \\
    --genesis-host genesis.peoplesainetwork.com \\
    --genesis-port 30300 \\
    --local-port 30301 \\
    --node-name csp-node \\
    --log-level INFO \\
    --max-connection-attempts 50

echo ""
echo "👋 Enhanced CSP Network shutdown complete"
"""
        
        script_path = Path("start_network.sh")
        try:
            with open(script_path, 'w') as f:
                f.write(script_content)
            os.chmod(script_path, 0o755)  # Make executable
            logger.info(f"✅ Startup script created: {script_path}")
            return str(script_path)
        except Exception as e:
            logger.error(f"❌ Failed to create startup script: {e}")
            return ""
    
    def setup_environment(self) -> Dict[str, str]:
        """Setup environment variables for the network."""
        env_vars = {
            "CSP_NETWORK_MODE": "genesis_connect",
            "CSP_LOG_LEVEL": "INFO",
            "CSP_DATA_DIR": "./network_data",
            "CSP_CONFIG_DIR": str(self.config_dir),
            "PYTHONPATH": f"{os.getcwd()}:{os.environ.get('PYTHONPATH', '')}"
        }
        
        # Set environment variables
        for key, value in env_vars.items():
            os.environ[key] = value
            logger.info(f"🔧 Set {key}={value}")
        
        return env_vars
    
    def check_dependencies(self) -> List[str]:
        """Check if required dependencies are available."""
        missing = []
        
        try:
            import asyncio
            import aiohttp
        except ImportError:
            missing.append("aiohttp")
        
        try:
            import cryptography
        except ImportError:
            missing.append("cryptography")
        
        # Check for enhanced_csp modules
        try:
            import enhanced_csp.network
        except ImportError:
            missing.append("enhanced_csp.network")
        
        return missing
    
    def print_status(self, config: Dict[str, Any]):
        """Print configuration status."""
        print("🔧 Enhanced CSP Network Configuration")
        print("=" * 50)
        
        if "network" in config:
            print(f"🏷️  Node Name: {config['network'].get('node_name', 'unknown')}")
            print(f"🌐 Network ID: {config['network'].get('network_id', 'unknown')}")
        
        if "p2p" in config:
            print(f"🔌 Local Port: {config['p2p'].get('listen_port', 'unknown')}")
            bootstrap_count = len(config['p2p'].get('bootstrap_nodes', []))
            print(f"🔗 Bootstrap Nodes: {bootstrap_count}")
        
        if "capabilities" in config:
            caps = config['capabilities']
            enabled_caps = [k for k, v in caps.items() if v]
            print(f"⚡ Capabilities: {', '.join(enabled_caps) if enabled_caps else 'none'}")
        
        print()


def main():
    """Main function for configuration setup."""
    helper = NetworkConfigHelper()
    
    print("🔧 Enhanced CSP Network Configuration Helper")
    print("=" * 50)
    
    # Check dependencies
    missing_deps = helper.check_dependencies()
    if missing_deps:
        print(f"⚠️  Missing dependencies: {', '.join(missing_deps)}")
        print("💡 Run: pip install -r requirements-lock.txt")
        print()
    
    # Create configuration
    config = helper.create_genesis_connection_config()
    
    # Validate configuration
    issues = helper.validate_config(config)
    if issues:
        print("⚠️  Configuration issues found:")
        for issue in issues:
            print(f"   • {issue}")
        print()
    else:
        print("✅ Configuration validation passed")
    
    # Save configuration
    config_path = helper.save_config(config)
    if config_path:
        print(f"💾 Configuration saved to: {config_path}")
    
    # Setup environment
    env_vars = helper.setup_environment()
    
    # Create startup script
    startup_script = helper.create_startup_script(config_path or "config/network_config.json")
    if startup_script:
        print(f"📜 Startup script created: {startup_script}")
    
    # Print status
    helper.print_status(config)
    
    print("🎉 Configuration setup complete!")
    print()
    print("Next steps:")
    print("1. Review the configuration in config/network_config.json")
    print("2. Run: ./start_network.sh")
    print("   OR")
    print("   Run: python3 network_startup.py")
    print()


if __name__ == "__main__":
    main()
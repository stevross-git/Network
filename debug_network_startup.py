#!/usr/bin/env python3
"""
Debug version of network startup that provides detailed error information.
This will help us identify exactly what's failing during network startup.
"""

import asyncio
import logging
import sys
import traceback
from pathlib import Path

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)8s] %(name)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Reduce noise from some modules
logging.getLogger('asyncio').setLevel(logging.INFO)

logger = logging.getLogger("debug_startup")

async def debug_network_startup():
    """Debug the network startup process with detailed error reporting."""
    
    print("🔍 Enhanced CSP Network - Detailed Debug Startup")
    print("=" * 65)
    
    try:
        # Import the network components
        print("\n📦 Importing network components...")
        from enhanced_csp.network.core.config import NetworkConfig, P2PConfig, MeshConfig
        from enhanced_csp.network.core.node import NetworkNode
        from enhanced_csp.network.core.network import EnhancedCSPNetwork
        
        print("✅ Imports successful")
        
        # Create configuration similar to the genesis connector
        print("\n🔧 Creating network configuration...")
        config = NetworkConfig()
        
        # P2P Configuration (similar to genesis_connector)
        config.p2p = P2PConfig()
        config.p2p.listen_address = "0.0.0.0"
        config.p2p.listen_port = 30301
        config.p2p.enable_mdns = True
        config.p2p.enable_upnp = True
        config.p2p.enable_nat_traversal = True
        
        # Genesis bootstrap
        config.p2p.bootstrap_nodes = ["/ip4/genesis.peoplesainetwork.com/tcp/30300"]
        config.p2p.dns_seed_domain = "peoplesainetwork.com"
        
        # Mesh configuration
        config.mesh = MeshConfig()
        config.mesh.enable_super_peers = True
        config.mesh.max_peers = 50
        config.mesh.topology_type = "dynamic_partial"
        
        # Node settings
        config.node_name = "csp-node"
        config.node_type = "standard"
        
        # Enable features
        config.enable_dht = True
        config.enable_mesh = True
        config.enable_dns = True
        config.enable_adaptive_routing = True
        
        # Data directory
        config.data_dir = Path("./network_data")
        config.data_dir.mkdir(exist_ok=True)
        
        print("✅ Configuration created")
        
        # Try creating EnhancedCSPNetwork (what genesis_connector uses)
        print("\n🌐 Creating EnhancedCSPNetwork...")
        
        try:
            network = EnhancedCSPNetwork(config)
            print("✅ EnhancedCSPNetwork created")
            
            print("\n🚀 Starting EnhancedCSPNetwork...")
            
            # This is where the failure likely occurs
            result = await network.start()
            print(f"✅ Network start result: {result}")
            
            if result:
                print("🎉 Network started successfully!")
                
                # Let it run briefly
                print("⏱️ Running for 5 seconds...")
                await asyncio.sleep(5)
                
                # Clean shutdown
                print("🛑 Shutting down network...")
                await network.stop()
                print("✅ Network shutdown completed")
                
                return True
            else:
                print("❌ Network start returned False")
                return False
                
        except Exception as e:
            print(f"❌ EnhancedCSPNetwork error: {e}")
            print("📋 Full traceback:")
            traceback.print_exc()
            
            # Try to get more details about the error
            print("\n🔍 Analyzing error details...")
            
            # Check if it's an import error
            if "import" in str(e).lower():
                print("   → Import error detected")
            
            # Check if it's a configuration error
            if "config" in str(e).lower():
                print("   → Configuration error detected")
                
            # Check if it's a network error
            if any(word in str(e).lower() for word in ["socket", "address", "port", "bind"]):
                print("   → Network/socket error detected")
                
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("📋 Checking what's available...")
        
        # Try importing components individually
        components = [
            ("NetworkConfig", "enhanced_csp.network.core.config", "NetworkConfig"),
            ("EnhancedCSPNetwork", "enhanced_csp.network.core.network", "EnhancedCSPNetwork"),
            ("NetworkNode", "enhanced_csp.network.core.node", "NetworkNode"),
        ]
        
        for name, module, cls in components:
            try:
                mod = __import__(module, fromlist=[cls])
                getattr(mod, cls)
                print(f"   ✅ {name} available")
            except Exception as import_err:
                print(f"   ❌ {name} not available: {import_err}")
        
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("📋 Full traceback:")
        traceback.print_exc()
        return False

async def debug_individual_components():
    """Test individual components to isolate the issue."""
    
    print("\n🧪 Testing individual components...")
    
    try:
        from enhanced_csp.network.core.config import NetworkConfig
        from enhanced_csp.network.core.node import NetworkNode
        
        # Test basic NetworkNode (like our simple test)
        print("\n🔬 Testing basic NetworkNode...")
        config = NetworkConfig()
        config.enable_dht = False
        config.enable_mesh = False
        config.enable_dns = False
        config.enable_adaptive_routing = False
        
        node = NetworkNode(config)
        print("✅ NetworkNode created")
        
        # Try starting the node
        result = await node.start()
        print(f"✅ NetworkNode start result: {result}")
        
        if result:
            await asyncio.sleep(1)
            await node.stop()
            print("✅ NetworkNode stopped")
            return True
        else:
            print("❌ NetworkNode failed to start")
            return False
            
    except Exception as e:
        print(f"❌ NetworkNode test failed: {e}")
        traceback.print_exc()
        return False

async def main():
    """Main debug entry point."""
    
    success1 = await debug_network_startup()
    
    if not success1:
        print("\n" + "="*50)
        print("🔄 Trying individual component test...")
        success2 = await debug_individual_components()
        
        if success2:
            print("\n💡 Individual components work, but EnhancedCSPNetwork fails")
            print("   → The issue is likely in the EnhancedCSPNetwork integration")
        else:
            print("\n💡 Even basic NetworkNode fails in this context")
            print("   → The issue might be in the configuration or environment")
    
    if success1:
        print("\n🎉 Network startup debugging completed successfully!")
        print("   → The network should work with the main startup script")
    else:
        print("\n🔍 Network startup debugging revealed issues")
        print("   → Check the error details above for the root cause")
    
    return success1

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

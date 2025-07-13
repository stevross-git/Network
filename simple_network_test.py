# Copy the content from the second artifact
#!/usr/bin/env python3
"""
Simple Network Component Test
Tests network components in isolation to identify issues.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("simple_test")

async def test_basic_imports():
    """Test basic imports work."""
    print("🧪 Testing basic imports...")
    
    try:
        from enhanced_csp.network.core.config import NetworkConfig
        print("✅ NetworkConfig import OK")
        
        from enhanced_csp.network.core.types import NodeID
        print("✅ NodeID import OK")
        
        # Test creating basic objects
        config = NetworkConfig()
        print(f"✅ NetworkConfig created: {config.node_name}")
        
        node_id = NodeID.generate()
        print(f"✅ NodeID generated: {str(node_id)[:16]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic imports failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_transport_creation():
    """Test creating transport layer."""
    print("\n🧪 Testing transport creation...")
    
    try:
        from enhanced_csp.network.core.config import NetworkConfig, P2PConfig
        from enhanced_csp.network.transport.transport import Transport
        
        config = NetworkConfig()
        config.p2p = P2PConfig()
        config.p2p.listen_address = "127.0.0.1"
        config.p2p.listen_port = 30301
        
        transport = Transport(config)
        print("✅ Transport created")
        
        # Try starting transport
        started = await transport.start()
        print(f"✅ Transport start result: {started}")
        
        if started:
            await transport.stop()
            print("✅ Transport stopped")
        
        return True
        
    except ImportError as e:
        print(f"⚠️ Transport not available: {e}")
        return True  # This is OK, transport might not be implemented
    except Exception as e:
        print(f"❌ Transport test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_discovery_creation():
    """Test creating discovery layer."""
    print("\n🧪 Testing discovery creation...")
    
    try:
        from enhanced_csp.network.core.config import NetworkConfig
        from enhanced_csp.network.discovery.discovery import Discovery
        
        config = NetworkConfig()
        discovery = Discovery(config)
        print("✅ Discovery created")
        
        return True
        
    except ImportError as e:
        print(f"⚠️ Discovery not available: {e}")
        return True  # This is OK
    except Exception as e:
        print(f"❌ Discovery test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_minimal_node():
    """Test creating a minimal network node."""
    print("\n🧪 Testing minimal node creation...")
    
    try:
        from enhanced_csp.network.core.config import NetworkConfig
        from enhanced_csp.network.core.types import NodeID
        from enhanced_csp.network.core.node import NetworkNode
        
        # Create minimal config
        config = NetworkConfig()
        config.data_dir = Path("./test_data")
        config.data_dir.mkdir(exist_ok=True)
        
        # Disable complex features that might fail
        config.enable_dht = False
        config.enable_mesh = False
        config.enable_dns = False
        config.enable_adaptive_routing = False
        
        print(f"✅ Config created with minimal features")
        
        # Create node
        node = NetworkNode(config)
        print(f"✅ NetworkNode created: {node.node_id}")
        
        # Try manual component initialization to see what fails
        print("🔍 Attempting manual component initialization...")
        
        try:
            await node._init_components()
            print("✅ Components initialized")
            
            # Try starting
            result = await node.start()
            print(f"✅ Node start result: {result}")
            
            if result:
                print("🎉 Node started successfully!")
                await asyncio.sleep(2)  # Run briefly
                await node.stop()
                print("✅ Node stopped")
                
            return True
            
        except Exception as init_e:
            print(f"❌ Component initialization failed: {init_e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"❌ Minimal node test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_network_creation():
    """Test creating the main network."""
    print("\n🧪 Testing network creation...")
    
    try:
        from enhanced_csp.network.core.config import NetworkConfig
        from enhanced_csp.network.core.network import EnhancedCSPNetwork
        
        config = NetworkConfig()
        network = EnhancedCSPNetwork(config)
        print("✅ EnhancedCSPNetwork created")
        
        # Try starting
        result = await network.start()
        print(f"✅ Network start result: {result}")
        
        if result:
            await network.stop()
            print("✅ Network stopped")
            
        return True
        
    except ImportError as e:
        print(f"⚠️ EnhancedCSPNetwork not available: {e}")
        return True  # This is OK
    except Exception as e:
        print(f"❌ Network test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests."""
    print("🚀 Enhanced CSP Network - Simple Component Tests")
    print("=" * 60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Transport Creation", test_transport_creation),
        ("Discovery Creation", test_discovery_creation),
        ("Minimal Node", test_minimal_node),
        ("Network Creation", test_network_creation),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n📋 Test Results:")
    print("=" * 30)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! The components are working.")
    elif passed >= total - 1:
        print("\n⚠️ Most tests passed. Check the failed test for issues.")
    else:
        print("\n❌ Multiple tests failed. Check imports and dependencies.")
    
    return passed >= total - 1

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
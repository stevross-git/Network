#!/usr/bin/env python3
"""
Genesis Server Connectivity Test
Quick test to verify connectivity to the Enhanced CSP Genesis server.
"""

import asyncio
import socket
import ssl
import sys
import time
from typing import Tuple, List
import aiohttp
import json

class GenesisConnectivityTest:
    """Test connectivity to the Enhanced CSP Genesis server."""
    
    def __init__(self):
        self.genesis_host = "genesis.peoplesainetwork.com"
        self.p2p_port = 30300      # P2P network port
        self.https_port = 443      # HTTPS/Status port
        self.status_port = 6969    # Legacy status port (now on HTTPS)
        
    async def test_all_connectivity(self):
        """Run all connectivity tests."""
        print("🔍 Enhanced CSP Genesis Server Connectivity Test")
        print("=" * 60)
        print(f"🌐 Target: {self.genesis_host}")
        print(f"📅 Test Date: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
        # Also test WebSocket connections (common for P2P)
        await self._test_websocket_connection()
        
    async def _test_websocket_connection(self):
        """Test WebSocket connection to genesis server."""
        print("🌐 Testing WebSocket Connection")
        print("-" * 30)
        
        websocket_urls = [
            f"wss://{self.genesis_host}/ws",
            f"wss://{self.genesis_host}/network", 
            f"wss://{self.genesis_host}/p2p",
            f"ws://{self.genesis_host}:30300/ws",
            f"ws://{self.genesis_host}:6969/ws"
        ]
        
        for url in websocket_urls:
            try:
                print(f"Testing {url}...", end=" ")
                
                # Try websocket connection
                import websockets
                async with websockets.connect(url, timeout=5) as websocket:
                    print("✅ WebSocket connected!")
                    break
                    
            except ImportError:
                print("❌ websockets library not available")
                break
            except Exception as e:
                print(f"❌ {e}")
                
        print()
        
        # Test P2P ports
        await self._test_p2p_port()
        
        # Test HTTPS/Status port
        await self._test_https_port()
        
        # Test HTTP API if available
        await self._test_status_api()
        
        # Test DNS resolution
        await self._test_dns_resolution()
        
        print("\n✅ Genesis connectivity test complete!")
        
        # Test multiple possible P2P ports
        p2p_ports_to_test = [30300, 4001, 8080, 6969, 9000, 5000]
        
        print(f"🔌 Testing P2P Ports on {self.genesis_host}")
        print("-" * 40)
        
        working_ports = []
        
        for port in p2p_ports_to_test:
            try:
                print(f"Testing port {port}...", end=" ")
                
                # Test TCP connection
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(self.genesis_host, port),
                    timeout=5
                )
                
                # Get connection info
                peername = writer.get_extra_info('peername')
                print(f"✅ Connected! ({peername[0]}:{peername[1]})")
                working_ports.append(port)
                
                # Close connection gracefully
                writer.close()
                await writer.wait_closed()
                
            except asyncio.TimeoutError:
                print("⏱️  Timeout")
                
            except ConnectionRefusedError:
                print("❌ Refused")
                
            except Exception as e:
                print(f"❌ Failed: {e}")
        
        if working_ports:
            print(f"\n🎉 Working ports found: {working_ports}")
            print(f"💡 Try using one of these ports instead of {self.p2p_port}")
        else:
            print(f"\n⚠️  No P2P ports are accessible")
            print("💡 The genesis server might be behind a firewall")
            print("💡 Or using a different protocol (WebSocket, HTTP, etc.)")
        
        print()
    
    async def _test_https_port(self):
        """Test the HTTPS port (443)."""
        print(f"🔒 Testing HTTPS Port ({self.https_port})")
        print("-" * 30)
        
        try:
            print(f"Connecting to https://{self.genesis_host}...", end=" ")
            
            # Test HTTPS connection
            context = ssl.create_default_context()
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(
                    self.genesis_host, 
                    self.https_port, 
                    ssl=context
                ),
                timeout=10
            )
            
            print("✅ SSL/TLS Connected!")
            
            # Get SSL certificate info
            ssl_object = writer.get_extra_info('ssl_object')
            if ssl_object:
                cert = ssl_object.getpeercert()
                print(f"   📜 Certificate Subject: {cert.get('subject', 'Unknown')}")
                print(f"   🏢 Certificate Issuer: {cert.get('issuer', 'Unknown')}")
                print(f"   📅 Valid Until: {cert.get('notAfter', 'Unknown')}")
            
            writer.close()
            await writer.wait_closed()
            
        except Exception as e:
            print(f"❌ HTTPS connection failed: {e}")
        
        print()
    
    async def _test_status_api(self):
        """Test the status API endpoint."""
        print("📊 Testing Status API")
        print("-" * 30)
        
        endpoints_to_test = [
            f"https://{self.genesis_host}/",
            f"https://{self.genesis_host}/status",
            f"https://{self.genesis_host}/health",
            f"https://{self.genesis_host}/api/status",
            f"https://{self.genesis_host}/network/status"
        ]
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            for endpoint in endpoints_to_test:
                try:
                    print(f"GET {endpoint}...", end=" ")
                    
                    async with session.get(endpoint) as response:
                        status_code = response.status
                        content_type = response.headers.get('content-type', 'unknown')
                        
                        if status_code == 200:
                            print(f"✅ {status_code} ({content_type})")
                            
                            # Try to read response
                            try:
                                if 'json' in content_type:
                                    data = await response.json()
                                    print(f"   📄 Response: {json.dumps(data, indent=2)[:200]}...")
                                else:
                                    text = await response.text()
                                    print(f"   📄 Response: {text[:100]}...")
                            except:
                                print("   📄 (Could not read response body)")
                                
                        elif status_code in [301, 302, 303, 307, 308]:
                            location = response.headers.get('location', 'unknown')
                            print(f"🔄 {status_code} → {location}")
                            
                        else:
                            print(f"⚠️  {status_code}")
                            
                except asyncio.TimeoutError:
                    print("⏱️  Timeout")
                    
                except Exception as e:
                    print(f"❌ Error: {e}")
        
        print()
    
    async def _test_dns_resolution(self):
        """Test DNS resolution for the genesis domain."""
        print("🔍 Testing DNS Resolution")
        print("-" * 30)
        
        domains_to_test = [
            self.genesis_host,
            "seed1.peoplesainetwork.com",
            "seed2.peoplesainetwork.com", 
            "bootstrap.peoplesainetwork.com"
        ]
        
        for domain in domains_to_test:
            try:
                print(f"Resolving {domain}...", end=" ")
                
                # DNS lookup
                loop = asyncio.get_event_loop()
                addrs = await loop.getaddrinfo(
                    domain, 
                    None, 
                    family=socket.AF_INET,
                    type=socket.SOCK_STREAM
                )
                
                ips = list(set([addr[4][0] for addr in addrs]))
                print(f"✅ {', '.join(ips)}")
                
            except Exception as e:
                print(f"❌ Failed: {e}")
        
        print()
    
    def print_connection_summary(self):
        """Print a summary of connection details."""
        print("📋 Connection Summary")
        print("=" * 60)
        print(f"🌐 Genesis Server: {self.genesis_host}")
        print(f"🔌 P2P Network Port: {self.p2p_port}")
        print(f"🔒 HTTPS Status Port: {self.https_port}")
        print(f"📊 Legacy Status Port: {self.status_port} (now via HTTPS)")
        print()
        print("🚀 To connect your node:")
        print(f"   python3 network_startup.py \\")
        print(f"     --genesis-host {self.genesis_host} \\")
        print(f"     --genesis-port {self.p2p_port} \\")
        print(f"     --local-port 30301")
        print()


async def main():
    """Main test function."""
    tester = GenesisConnectivityTest()
    
    # Print summary first
    tester.print_connection_summary()
    
    # Run connectivity tests
    await tester.test_all_connectivity()
    
    print("💡 Next Steps:")
    print("   1. If P2P port (30300) is accessible: ✅ Ready to connect!")
    print("   2. If HTTPS (443) works: ✅ Status monitoring available!")
    print("   3. Run: python3 network_startup.py --quick-start")
    print()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
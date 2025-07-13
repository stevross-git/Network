#!/usr/bin/env python3
"""
Enhanced CSP Network Diagnostics Tool
Diagnoses network connectivity issues and provides solutions.
"""

import asyncio
import socket
import sys
import time
import json
import subprocess
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


class NetworkDiagnostics:
    """Comprehensive network diagnostics for Enhanced CSP."""
    
    def __init__(self):
        self.genesis_hosts = [
            ("genesis.peoplesainetwork.com", 30300),
            ("147.75.77.187", 30300),  # Backup IP
            ("seed1.peoplesainetwork.com", 30300),
            ("seed2.peoplesainetwork.com", 30300),
            ("bootstrap.peoplesainetwork.com", 30300)
        ]
        self.results: Dict[str, any] = {}
        
    async def run_full_diagnostics(self) -> Dict[str, any]:
        """Run complete network diagnostics."""
        print("🔍 Enhanced CSP Network Diagnostics")
        print("=" * 50)
        
        # System checks
        await self._check_system_requirements()
        
        # Network connectivity
        await self._check_network_connectivity()
        
        # DNS resolution
        await self._check_dns_resolution()
        
        # Port availability
        await self._check_port_availability()
        
        # Python environment
        await self._check_python_environment()
        
        # Enhanced CSP modules
        await self._check_csp_modules()
        
        # Configuration files
        await self._check_configuration()
        
        # Generate report
        self._generate_report()
        
        return self.results
    
    async def _check_system_requirements(self):
        """Check system requirements."""
        print("\n🖥️  System Requirements Check")
        print("-" * 30)
        
        # Operating system
        os_info = platform.system()
        os_version = platform.release()
        print(f"OS: {os_info} {os_version}")
        
        # Python version
        python_version = sys.version.split()[0]
        python_ok = sys.version_info >= (3, 8)
        status = "✅" if python_ok else "❌"
        print(f"Python: {python_version} {status}")
        
        # Memory check
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            memory_ok = memory_gb >= 1.0  # Minimum 1GB
            status = "✅" if memory_ok else "⚠️"
            print(f"Memory: {memory_gb:.1f} GB {status}")
        except ImportError:
            print("Memory: Cannot check (psutil not available)")
        
        # Disk space
        try:
            disk_usage = Path.cwd().stat()
            print("Disk: Available ✅")
        except:
            print("Disk: Cannot check ⚠️")
        
        self.results["system"] = {
            "os": os_info,
            "python_version": python_version,
            "python_ok": python_ok
        }
    
    async def _check_network_connectivity(self):
        """Check network connectivity to genesis servers."""
        print("\n🌐 Network Connectivity Check")
        print("-" * 30)
        
        connectivity_results = []
        
        for host, port in self.genesis_hosts:
            print(f"Testing {host}:{port}...", end=" ")
            
            try:
                # TCP connection test
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(host, port),
                    timeout=10
                )
                writer.close()
                await writer.wait_closed()
                print("✅ Connected")
                connectivity_results.append((host, port, True, "Connected"))
                
            except asyncio.TimeoutError:
                print("⏱️  Timeout")
                connectivity_results.append((host, port, False, "Timeout"))
                
            except Exception as e:
                print(f"❌ Failed: {e}")
                connectivity_results.append((host, port, False, str(e)))
        
        # Check internet connectivity
        print("\nInternet connectivity...", end=" ")
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection("8.8.8.8", 53),  # Google DNS
                timeout=5
            )
            writer.close()
            await writer.wait_closed()
            print("✅ Internet OK")
            internet_ok = True
        except:
            print("❌ No internet")
            internet_ok = False
        
        self.results["connectivity"] = {
            "genesis_servers": connectivity_results,
            "internet": internet_ok
        }
    
    async def _check_dns_resolution(self):
        """Check DNS resolution for genesis hosts."""
        print("\n🔍 DNS Resolution Check")
        print("-" * 30)
        
        dns_results = []
        
        for host, port in self.genesis_hosts:
            if host.replace(".", "").isdigit():  # Skip IP addresses
                continue
                
            print(f"Resolving {host}...", end=" ")
            
            try:
                # DNS lookup
                loop = asyncio.get_event_loop()
                addrs = await loop.getaddrinfo(host, port, family=socket.AF_INET)
                ips = [addr[4][0] for addr in addrs]
                print(f"✅ {', '.join(set(ips))}")
                dns_results.append((host, True, ips))
                
            except Exception as e:
                print(f"❌ Failed: {e}")
                dns_results.append((host, False, str(e)))
        
        self.results["dns"] = dns_results
    
    async def _check_port_availability(self):
        """Check if local ports are available."""
        print("\n🔌 Port Availability Check")
        print("-" * 30)
        
        test_ports = [30301, 30302, 30303, 4002, 4003]
        port_results = []
        
        for port in test_ports:
            print(f"Testing port {port}...", end=" ")
            
            try:
                # Try to bind to the port
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(('0.0.0.0', port))
                sock.close()
                print("✅ Available")
                port_results.append((port, True, "Available"))
                
            except OSError as e:
                print(f"❌ In use")
                port_results.append((port, False, "In use"))
            except Exception as e:
                print(f"⚠️  Error: {e}")
                port_results.append((port, False, str(e)))
        
        self.results["ports"] = port_results
    
    async def _check_python_environment(self):
        """Check Python environment and dependencies."""
        print("\n🐍 Python Environment Check")
        print("-" * 30)
        
        # Required packages
        required_packages = [
            "asyncio",
            "aiohttp", 
            "cryptography",
            "pathlib",
            "json",
            "socket"
        ]
        
        package_results = []
        
        for package in required_packages:
            print(f"Checking {package}...", end=" ")
            
            try:
                __import__(package)
                print("✅ Available")
                package_results.append((package, True, "Available"))
            except ImportError:
                print("❌ Missing")
                package_results.append((package, False, "Missing"))
        
        # Check pip
        print("Checking pip...", end=" ")
        try:
            import pip
            print("✅ Available")
            pip_available = True
        except ImportError:
            print("❌ Missing")
            pip_available = False
        
        self.results["python_env"] = {
            "packages": package_results,
            "pip_available": pip_available
        }
    
    async def _check_csp_modules(self):
        """Check Enhanced CSP module availability."""
        print("\n🔧 Enhanced CSP Modules Check")
        print("-" * 30)
        
        # Add current directory to Python path
        sys.path.insert(0, str(Path.cwd()))
        
        csp_modules = [
            "enhanced_csp",
            "enhanced_csp.network",
            "enhanced_csp.network.core",
            "enhanced_csp.network.core.config",
            "enhanced_csp.network.core.node",
            "enhanced_csp.network.utils"
        ]
        
        module_results = []
        
        for module in csp_modules:
            print(f"Checking {module}...", end=" ")
            
            try:
                __import__(module)
                print("✅ Available")
                module_results.append((module, True, "Available"))
            except ImportError as e:
                print(f"❌ Missing: {e}")
                module_results.append((module, False, str(e)))
        
        self.results["csp_modules"] = module_results
    
    async def _check_configuration(self):
        """Check configuration files and directories."""
        print("\n📋 Configuration Check")
        print("-" * 30)
        
        # Check important files and directories
        paths_to_check = [
            ("enhanced_csp/", "Enhanced CSP directory"),
            ("enhanced_csp/network/", "Network module directory"),
            ("enhanced_csp/run_network.py", "Network runner script"),
            ("config/", "Configuration directory"),
            ("network_data/", "Network data directory"),
            ("requirements-lock.txt", "Requirements file")
        ]
        
        config_results = []
        
        for path_str, description in paths_to_check:
            path = Path(path_str)
            print(f"Checking {description}...", end=" ")
            
            if path.exists():
                if path.is_dir():
                    print("✅ Directory exists")
                else:
                    print("✅ File exists")
                config_results.append((path_str, True, "Exists"))
            else:
                print("❌ Missing")
                config_results.append((path_str, False, "Missing"))
        
        self.results["configuration"] = config_results
    
    def _generate_report(self):
        """Generate diagnostic report with recommendations."""
        print("\n📊 Diagnostic Report")
        print("=" * 50)
        
        # Count issues
        total_checks = 0
        passed_checks = 0
        issues = []
        
        # System issues
        if not self.results.get("system", {}).get("python_ok", False):
            issues.append("❌ Python version < 3.8 - upgrade to Python 3.8+")
        
        # Connectivity issues
        connectivity = self.results.get("connectivity", {})
        genesis_servers = connectivity.get("genesis_servers", [])
        connected_servers = [s for s in genesis_servers if s[2]]
        
        if not connected_servers:
            issues.append("❌ Cannot connect to any genesis servers")
        elif len(connected_servers) < len(genesis_servers):
            issues.append("⚠️  Some genesis servers unreachable")
        
        if not connectivity.get("internet", False):
            issues.append("❌ No internet connectivity")
        
        # DNS issues
        dns_results = self.results.get("dns", [])
        failed_dns = [d for d in dns_results if not d[1]]
        if failed_dns:
            issues.append(f"❌ DNS resolution failed for: {', '.join([d[0] for d in failed_dns])}")
        
        # Port issues
        port_results = self.results.get("ports", [])
        unavailable_ports = [p for p in port_results if not p[1]]
        if unavailable_ports:
            issues.append(f"⚠️  Ports in use: {', '.join([str(p[0]) for p in unavailable_ports])}")
        
        # Python environment issues
        python_env = self.results.get("python_env", {})
        missing_packages = [p for p in python_env.get("packages", []) if not p[1]]
        if missing_packages:
            issues.append(f"❌ Missing packages: {', '.join([p[0] for p in missing_packages])}")
        
        # CSP module issues
        csp_modules = self.results.get("csp_modules", [])
        missing_modules = [m for m in csp_modules if not m[1]]
        if missing_modules:
            issues.append(f"❌ Missing CSP modules: {', '.join([m[0] for m in missing_modules])}")
        
        # Configuration issues
        config_results = self.results.get("configuration", [])
        missing_config = [c for c in config_results if not c[1]]
        if missing_config:
            issues.append(f"❌ Missing files/directories: {', '.join([c[0] for c in missing_config])}")
        
        # Print summary
        if not issues:
            print("🎉 All checks passed! Network should work correctly.")
        else:
            print(f"Found {len(issues)} issues:")
            for issue in issues:
                print(f"  {issue}")
        
        print("\n💡 Recommendations:")
        self._print_recommendations(issues)
        
        # Save results
        self._save_results()
    
    def _print_recommendations(self, issues: List[str]):
        """Print specific recommendations based on issues found."""
        
        if any("Python version" in issue for issue in issues):
            print("  • Install Python 3.8 or later")
            
        if any("Cannot connect" in issue for issue in issues):
            print("  • Check your internet connection")
            print("  • Check firewall settings")
            print("  • Try using --quick-start flag to skip connectivity checks")
            
        if any("Missing packages" in issue for issue in issues):
            print("  • Run: pip install -r requirements-lock.txt")
            print("  • Or: pip install aiohttp cryptography")
            
        if any("Missing CSP modules" in issue for issue in issues):
            print("  • Ensure you're in the correct project directory")
            print("  • Check that enhanced_csp/ directory exists")
            print("  • Run from project root directory")
            
        if any("Ports in use" in issue for issue in issues):
            print("  • Use different ports with --local-port flag")
            print("  • Kill processes using those ports")
            
        if any("DNS resolution" in issue for issue in issues):
            print("  • Check DNS settings")
            print("  • Try using IP addresses instead of hostnames")
            
        print("  • Try the simple startup: python3 network_startup.py --quick-start")
        print("  • Check logs for detailed error messages")
    
    def _save_results(self):
        """Save diagnostic results to file."""
        results_file = Path("network_diagnostics.json")
        
        try:
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"\n💾 Diagnostic results saved to: {results_file}")
        except Exception as e:
            print(f"\n⚠️  Could not save results: {e}")


async def main():
    """Main diagnostics function."""
    diagnostics = NetworkDiagnostics()
    await diagnostics.run_full_diagnostics()


if __name__ == "__main__":
    asyncio.run(main())
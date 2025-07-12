#!/usr/bin/env python3
"""Enhanced CSP System - Simple Main Application"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add current directory to Python path for imports
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import FastAPI
try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    logger.error("FastAPI not available. Install with: pip install fastapi uvicorn")
    FASTAPI_AVAILABLE = False

# Import our CSP components
try:
    from enhanced_csp.network.core.config import NetworkConfig
    logger.info("✅ Successfully imported NetworkConfig")
    CSP_AVAILABLE = True
except ImportError as e:
    logger.warning(f"CSP components not available: {e}")
    CSP_AVAILABLE = False

class SimpleCSPSystem:
    """Simple CSP system for demonstration."""
    
    def __init__(self):
        self.startup_time = datetime.now()
        self.config = None
        self.running = False
        
    async def initialize(self):
        """Initialize the CSP system."""
        logger.info("🚀 Initializing Simple CSP System...")
        
        if CSP_AVAILABLE:
            try:
                self.config = NetworkConfig()
                logger.info(f"✅ Created NetworkConfig: {self.config.node_name}")
            except Exception as e:
                logger.error(f"Failed to create config: {e}")
                self.config = None
        
        self.running = True
        logger.info("✅ Simple CSP System initialized")
    
    async def shutdown(self):
        """Shutdown the CSP system."""
        logger.info("🛑 Shutting down Simple CSP System...")
        self.running = False
        logger.info("✅ Simple CSP System shutdown complete")
    
    def get_status(self):
        """Get system status."""
        return {
            "status": "running" if self.running else "stopped",
            "startup_time": self.startup_time.isoformat(),
            "uptime": str(datetime.now() - self.startup_time),
            "config_available": self.config is not None,
            "csp_available": CSP_AVAILABLE,
            "fastapi_available": FASTAPI_AVAILABLE,
        }

# Global system instance
csp_system = SimpleCSPSystem()

def create_app():
    """Create FastAPI application."""
    if not FASTAPI_AVAILABLE:
        logger.error("Cannot create app without FastAPI")
        return None
    
    app = FastAPI(
        title="Enhanced CSP System",
        description="Simple CSP System Demo", 
        version="1.0.0"
    )
    
    # Add CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.on_event("startup")
    async def startup_event():
        await csp_system.initialize()
    
    @app.on_event("shutdown") 
    async def shutdown_event():
        await csp_system.shutdown()
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "Enhanced CSP System - Simple Version",
            "status": "running",
            **csp_system.get_status()
        }
    
    @app.get("/health")
    async def health():
        """Health check."""
        return {"status": "healthy", **csp_system.get_status()}
    
    @app.get("/status")
    async def status():
        """Detailed status."""
        return csp_system.get_status()
    
    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard():
        """Simple dashboard."""
        status = csp_system.get_status()
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Enhanced CSP System - Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
                .status {{ padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .status.good {{ background: #d4edda; color: #155724; }}
                .status.warning {{ background: #fff3cd; color: #856404; }}
                .status.error {{ background: #f8d7da; color: #721c24; }}
                h1 {{ color: #333; }}
                .metric {{ margin: 10px 0; padding: 10px; background: #f8f9fa; border-left: 4px solid #007bff; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 Enhanced CSP System Dashboard</h1>
                
                <div class="status {'good' if status['status'] == 'running' else 'error'}">
                    <strong>System Status:</strong> {status['status'].upper()}
                </div>
                
                <div class="metric">
                    <strong>Uptime:</strong> {status['uptime']}
                </div>
                
                <div class="metric">
                    <strong>Started:</strong> {status['startup_time']}
                </div>
                
                <div class="status {'good' if status['csp_available'] else 'warning'}">
                    <strong>CSP Core:</strong> {'✅ Available' if status['csp_available'] else '⚠️ Not Available'}
                </div>
                
                <div class="status {'good' if status['config_available'] else 'warning'}">
                    <strong>Configuration:</strong> {'✅ Loaded' if status['config_available'] else '⚠️ Not Loaded'}
                </div>
                
                <div class="status {'good' if status['fastapi_available'] else 'error'}">
                    <strong>FastAPI:</strong> {'✅ Available' if status['fastapi_available'] else '❌ Not Available'}
                </div>
                
                <h2>API Endpoints</h2>
                <ul>
                    <li><a href="/">Root</a> - Basic system info</li>
                    <li><a href="/health">Health</a> - Health check</li>
                    <li><a href="/status">Status</a> - Detailed status</li>
                    <li><a href="/docs">Docs</a> - API documentation</li>
                </ul>
                
                <h2>Test CSP Functionality</h2>
                <div id="test-results"></div>
                <button onclick="testCSP()">Test CSP Components</button>
                
                <script>
                async function testCSP() {{
                    const results = document.getElementById('test-results');
                    results.innerHTML = '<p>Testing...</p>';
                    
                    try {{
                        const response = await fetch('/test-csp');
                        const data = await response.json();
                        
                        let html = '<h3>Test Results:</h3><ul>';
                        for (const [test, result] of Object.entries(data.results)) {{
                            const status = result.success ? '✅' : '❌';
                            html += `<li>${{status}} ${{test}}: ${{result.message}}</li>`;
                        }}
                        html += '</ul>';
                        
                        results.innerHTML = html;
                    }} catch (error) {{
                        results.innerHTML = '<p>❌ Test failed: ' + error.message + '</p>';
                    }}
                }}
                </script>
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    @app.get("/test-csp")
    async def test_csp():
        """Test CSP functionality."""
        results = {}
        
        # Test 1: Config creation
        try:
            if CSP_AVAILABLE:
                from enhanced_csp.network.core.config import NetworkConfig
                config = NetworkConfig()
                results["config_creation"] = {
                    "success": True,
                    "message": f"Created config with node_name: {config.node_name}"
                }
            else:
                results["config_creation"] = {
                    "success": False,
                    "message": "CSP not available"
                }
        except Exception as e:
            results["config_creation"] = {
                "success": False,
                "message": str(e)
            }
        
        # Test 2: Types creation
        try:
            if CSP_AVAILABLE:
                from enhanced_csp.network.core.types import NodeID, MessageType
                node_id = NodeID.generate()
                results["types_creation"] = {
                    "success": True,
                    "message": f"Created NodeID: {str(node_id)[:20]}..."
                }
            else:
                results["types_creation"] = {
                    "success": False,
                    "message": "CSP not available"
                }
        except Exception as e:
            results["types_creation"] = {
                "success": False,
                "message": str(e)
            }
        
        # Test 3: Security components
        try:
            if CSP_AVAILABLE:
                from enhanced_csp.network.security.security_hardening import validate_ip_address
                valid_ip = validate_ip_address("192.168.1.1")
                results["security_validation"] = {
                    "success": True,
                    "message": f"IP validation works: {valid_ip}"
                }
            else:
                results["security_validation"] = {
                    "success": False,
                    "message": "CSP not available"
                }
        except Exception as e:
            results["security_validation"] = {
                "success": False,
                "message": str(e)
            }
        
        return {"results": results}
    
    return app

async def run_without_fastapi():
    """Run CSP system without FastAPI."""
    logger.info("🚀 Starting Enhanced CSP System (Console Mode)")
    
    await csp_system.initialize()
    
    try:
        logger.info("✅ System is running. Press Ctrl+C to stop.")
        logger.info(f"📊 System status: {csp_system.get_status()}")
        
        if CSP_AVAILABLE:
            # Test CSP functionality
            try:
                from enhanced_csp.network.core.config import NetworkConfig
                from enhanced_csp.network.core.types import NodeID
                
                config = NetworkConfig()
                node_id = NodeID.generate()
                
                logger.info(f"✅ CSP Test successful:")
                logger.info(f"   - Config: {config.node_name}")
                logger.info(f"   - NodeID: {str(node_id)[:30]}...")
                
            except Exception as e:
                logger.error(f"❌ CSP Test failed: {e}")
        
        # Keep running
        while True:
            await asyncio.sleep(10)
            logger.info(f"⏰ System running for {datetime.now() - csp_system.startup_time}")
            
    except KeyboardInterrupt:
        logger.info("🛑 Received shutdown signal")
    finally:
        await csp_system.shutdown()

def main():
    """Main entry point."""
    if FASTAPI_AVAILABLE:
        # Run with FastAPI
        app = create_app()
        if app:
            logger.info("🌐 Starting Enhanced CSP System with FastAPI...")
            logger.info("📍 Access dashboard at: http://localhost:8000/dashboard")
            uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
        else:
            logger.error("Failed to create FastAPI app")
    else:
        # Run without FastAPI
        asyncio.run(run_without_fastapi())

if __name__ == "__main__":
    main()
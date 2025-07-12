#!/usr/bin/env python3
"""Enhanced CSP System - Main Application Entry Point"""

import asyncio
import logging
import uuid
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add current directory to Python path
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

# FastAPI imports
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

# Pydantic for data validation
from pydantic import BaseModel, Field

# Import core CSP components with proper error handling
try:
    from enhanced_csp.network.core.config import NetworkConfig
    CSP_CORE_AVAILABLE = True
    logging.info("✅ Successfully imported NetworkConfig")
except ImportError as e:
    logging.warning(f"Core CSP components not available: {e}")
    CSP_CORE_AVAILABLE = False
    
    # Fallback implementations for development
    class NetworkConfig:
        def __init__(self, **kwargs):
            self.version = "1.0.0"
            self.environment = "development"

# Try to import EnhancedCSPNetwork (may not exist yet)
try:
    from enhanced_csp.network.core.node import EnhancedCSPNetwork
    NETWORK_NODE_AVAILABLE = True
except ImportError:
    logging.warning("EnhancedCSPNetwork not available, using fallback")
    NETWORK_NODE_AVAILABLE = False
    
    class EnhancedCSPNetwork:
        def __init__(self, config=None):
            self.config = config or NetworkConfig()
        
        async def start(self):
            logging.info("Enhanced CSP Network started (fallback mode)")
        
        async def stop(self):
            logging.info("Enhanced CSP Network stopped")

# Import AI components with fallbacks
try:
    from enhanced_csp.ai.reasoning_engine import MultiAgentReasoningCoordinator
    from enhanced_csp.ai.capabilities import LLMCapability, AIAgent
    AI_COMPONENTS_AVAILABLE = True
except ImportError:
    logging.warning("AI components not available")
    AI_COMPONENTS_AVAILABLE = False
    
    class MultiAgentReasoningCoordinator:
        def __init__(self, agents):
            self.agents = agents
    
    class LLMCapability:
        def __init__(self, model_name, specialized_domain=None):
            self.model_name = model_name
            self.specialized_domain = specialized_domain
    
    class AIAgent:
        def __init__(self, name, capabilities=None):
            self.name = name
            self.capabilities = capabilities or []

# Import optional components with proper fallbacks
RUNTIME_AVAILABLE = False
DEPLOYMENT_AVAILABLE = False 
DEV_TOOLS_AVAILABLE = False

class CSPRuntimeOrchestrator:
    def __init__(self, config):
        self.config = config
        self.running = False
    
    async def start(self):
        self.running = True
        logging.info("Runtime orchestrator started (fallback mode)")
    
    async def stop(self):
        self.running = False

class CSPDeploymentOrchestrator:
    def __init__(self):
        pass

class CSPDevelopmentTools:
    def __init__(self):
        self.visual_designer = self.CSPVisualDesigner()
        self.code_generator = self.CSPCodeGenerator()
    
    class CSPVisualDesigner:
        async def get_state(self):
            return {"status": "available", "components": []}
    
    class CSPCodeGenerator:
        async def generate_from_design(self, design_data):
            return "# Generated CSP code\nprint('Hello, CSP!')"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic models for API requests
class CollaborationRequest(BaseModel):
    agents: list = Field(default_factory=list)
    description: Optional[str] = None

class SystemMetrics(BaseModel):
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    active_connections: int = 0
    message_throughput: float = 0.0

class SystemState:
    """Global system state management."""
    
    def __init__(self):
        self.startup_time = datetime.now()
        self.network: Optional[EnhancedCSPNetwork] = None
        self.runtime: Optional[CSPRuntimeOrchestrator] = None
        self.deployment: Optional[CSPDeploymentOrchestrator] = None
        self.dev_tools: Optional[CSPDevelopmentTools] = None
        self.ai_engine = None
        self.active_processes: Dict[str, Any] = {}
        self.metrics = SystemMetrics()

# Global system state
system_state = SystemState()

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced CSP System",
    description="Advanced Computing Systems Platform with AI, Quantum, and Distributed Computing",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Mount static files
static_dir = Path("frontend/static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.on_event("startup")
async def startup_event():
    """Initialize system components on startup."""
    logger.info("🚀 Starting Enhanced CSP System...")
    
    try:
        # Initialize configuration
        config = NetworkConfig()
        
        # Initialize network
        if CSP_CORE_AVAILABLE and NETWORK_NODE_AVAILABLE:
            system_state.network = EnhancedCSPNetwork(config)
            await system_state.network.start()
        elif CSP_CORE_AVAILABLE:
            # Use fallback network
            system_state.network = EnhancedCSPNetwork(config)
            await system_state.network.start()
            logging.info("Using fallback network implementation")
        
        # Initialize runtime
        if RUNTIME_AVAILABLE:
            system_state.runtime = CSPRuntimeOrchestrator(config)
            await system_state.runtime.start()
        
        # Initialize deployment system
        if DEPLOYMENT_AVAILABLE:
            system_state.deployment = CSPDeploymentOrchestrator()
        
        # Initialize development tools
        if DEV_TOOLS_AVAILABLE:
            system_state.dev_tools = CSPDevelopmentTools()
        
        logger.info("✅ Enhanced CSP System started successfully")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown of system components."""
    logger.info("🛑 Shutting down Enhanced CSP System...")
    
    try:
        if system_state.runtime:
            await system_state.runtime.stop()
        
        if system_state.network:
            await system_state.network.stop()
        
        logger.info("✅ Enhanced CSP System shut down successfully")
        
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with system information."""
    return {
        "message": "Enhanced CSP System",
        "version": "1.0.0",
        "status": "running",
        "startup_time": system_state.startup_time.isoformat(),
        "components": {
            "network": system_state.network is not None,
            "runtime": system_state.runtime is not None,
            "deployment": DEPLOYMENT_AVAILABLE,
            "dev_tools": DEV_TOOLS_AVAILABLE,
            "ai_components": AI_COMPONENTS_AVAILABLE,
            "csp_core": CSP_CORE_AVAILABLE,
            "network_node": NETWORK_NODE_AVAILABLE,
        }
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime": str(datetime.now() - system_state.startup_time),
        "components_available": {
            "csp_core": CSP_CORE_AVAILABLE,
            "ai_components": AI_COMPONENTS_AVAILABLE,
            "runtime": RUNTIME_AVAILABLE,
            "deployment": DEPLOYMENT_AVAILABLE,
            "dev_tools": DEV_TOOLS_AVAILABLE,
        }
    }

# System information endpoint
@app.get("/api/system/info")
async def get_system_info():
    """Get detailed system information."""
    return {
        "version": "1.0.0",
        "environment": "development",
        "startup_time": system_state.startup_time.isoformat(),
        "uptime": str(datetime.now() - system_state.startup_time),
        "components": {
            "csp_core_available": CSP_CORE_AVAILABLE,
            "ai_components_available": AI_COMPONENTS_AVAILABLE,
            "runtime_available": RUNTIME_AVAILABLE,
            "deployment_available": DEPLOYMENT_AVAILABLE,
            "dev_tools_available": DEV_TOOLS_AVAILABLE,
        },
        "active_processes": len(system_state.active_processes),
        "metrics": system_state.metrics.dict()
    }

# System metrics endpoint
@app.get("/api/system/metrics")
async def get_system_metrics():
    """Get current system metrics."""
    try:
        import psutil
        
        # Update metrics
        system_state.metrics.cpu_usage = psutil.cpu_percent()
        system_state.metrics.memory_usage = psutil.virtual_memory().percent
        system_state.metrics.active_connections = len(system_state.active_processes)
        
    except ImportError:
        # Fallback metrics
        system_state.metrics.cpu_usage = 0.0
        system_state.metrics.memory_usage = 0.0
    
    return system_state.metrics.dict()

# AI collaboration endpoint
@app.post("/api/ai/collaboration")
async def create_ai_collaboration(request: CollaborationRequest):
    """Create an AI collaboration session."""
    if not AI_COMPONENTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="AI components not available")
    
    try:
        # Create AI agents
        agents = []
        for agent_spec in request.agents:
            capability = LLMCapability(
                model_name=agent_spec.get("model", "gpt-3.5-turbo"),
                specialized_domain=agent_spec.get("domain")
            )
            agent = AIAgent(
                name=agent_spec.get("name"),
                capabilities=[capability]
            )
            agents.append(agent)
        
        # Create collaborative process
        coordinator = MultiAgentReasoningCoordinator(agents)
        collaboration_id = str(uuid.uuid4())
        
        # Store collaboration
        system_state.active_processes[collaboration_id] = coordinator
        
        return {
            "collaboration_id": collaboration_id,
            "agents": [{"name": a.name, "capabilities": len(a.capabilities)} for a in agents],
            "status": "created"
        }
        
    except Exception as e:
        logger.error(f"AI collaboration creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Network status endpoint
@app.get("/api/network/status")
async def get_network_status():
    """Get network status information."""
    if not system_state.network:
        return {"status": "unavailable", "message": "Network not initialized"}
    
    return {
        "status": "running",
        "config": {
            "version": system_state.network.config.version,
            "environment": system_state.network.config.environment,
        },
        "connections": len(system_state.active_processes),
        "uptime": str(datetime.now() - system_state.startup_time)
    }

# Development tools endpoint
@app.get("/api/dev-tools/status")
async def get_dev_tools_status():
    """Get development tools status."""
    if not DEV_TOOLS_AVAILABLE or not system_state.dev_tools:
        return {"status": "unavailable", "message": "Development tools not available"}
    
    try:
        designer_state = await system_state.dev_tools.visual_designer.get_state()
        return {
            "status": "available",
            "visual_designer": designer_state,
            "code_generator": {"status": "ready"}
        }
    except Exception as e:
        logger.error(f"Dev tools status check failed: {e}")
        return {"status": "error", "message": str(e)}

# Code generation endpoint
@app.post("/api/dev-tools/generate")
async def generate_code(design_data: Dict[str, Any]):
    """Generate code from design data."""
    if not DEV_TOOLS_AVAILABLE or not system_state.dev_tools:
        raise HTTPException(status_code=503, detail="Development tools not available")
    
    try:
        generated_code = await system_state.dev_tools.code_generator.generate_from_design(design_data)
        return {
            "generated_code": generated_code,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Code generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Process management endpoints
@app.get("/api/processes")
async def list_processes():
    """List active processes."""
    return {
        "processes": list(system_state.active_processes.keys()),
        "count": len(system_state.active_processes)
    }

@app.delete("/api/processes/{process_id}")
async def stop_process(process_id: str):
    """Stop a specific process."""
    if process_id not in system_state.active_processes:
        raise HTTPException(status_code=404, detail="Process not found")
    
    try:
        process = system_state.active_processes[process_id]
        if hasattr(process, 'stop'):
            await process.stop()
        del system_state.active_processes[process_id]
        
        return {"message": f"Process {process_id} stopped successfully"}
    except Exception as e:
        logger.error(f"Failed to stop process {process_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Runtime orchestrator endpoints
@app.get("/api/runtime/status")
async def get_runtime_status():
    """Get runtime orchestrator status."""
    if not system_state.runtime:
        return {"status": "unavailable", "message": "Runtime not initialized"}
    
    return {
        "status": "running" if system_state.runtime.running else "stopped",
        "config": {
            "environment": getattr(system_state.runtime.config, 'environment', 'unknown')
        }
    }

@app.post("/api/runtime/restart")
async def restart_runtime():
    """Restart the runtime orchestrator."""
    if not system_state.runtime:
        raise HTTPException(status_code=503, detail="Runtime not available")
    
    try:
        await system_state.runtime.stop()
        await system_state.runtime.start()
        return {"message": "Runtime restarted successfully"}
    except Exception as e:
        logger.error(f"Runtime restart failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Deployment endpoints
@app.get("/api/deployment/status")
async def get_deployment_status():
    """Get deployment system status."""
    if not DEPLOYMENT_AVAILABLE or not system_state.deployment:
        return {"status": "unavailable", "message": "Deployment system not available"}
    
    return {
        "status": "available",
        "deployments": []  # Would contain actual deployment info
    }

# Configuration endpoints
@app.get("/api/config")
async def get_configuration():
    """Get current system configuration."""
    if system_state.network and hasattr(system_state.network, 'config'):
        return {
            "network": {
                "version": getattr(system_state.network.config, 'version', '1.0.0'),
                "environment": getattr(system_state.network.config, 'environment', 'development')
            }
        }
    
    return {
        "network": {
            "version": "1.0.0",
            "environment": "development"
        }
    }

@app.post("/api/config/update")
async def update_configuration(config_data: Dict[str, Any]):
    """Update system configuration."""
    try:
        # In a real implementation, this would update the actual configuration
        logger.info(f"Configuration update requested: {config_data}")
        return {"message": "Configuration updated successfully", "config": config_data}
    except Exception as e:
        logger.error(f"Configuration update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Testing endpoints
@app.get("/api/test/import-status")
async def get_import_status():
    """Get the status of module imports."""
    return {
        "csp_core": CSP_CORE_AVAILABLE,
        "ai_components": AI_COMPONENTS_AVAILABLE,
        "runtime": RUNTIME_AVAILABLE,
        "deployment": DEPLOYMENT_AVAILABLE,
        "dev_tools": DEV_TOOLS_AVAILABLE,
    }

@app.post("/api/test/run-diagnostics")
async def run_diagnostics():
    """Run system diagnostics."""
    diagnostics = {
        "timestamp": datetime.now().isoformat(),
        "system_info": {
            "python_version": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
            "platform": __import__('platform').system(),
        },
        "component_status": {
            "csp_core": CSP_CORE_AVAILABLE,
            "ai_components": AI_COMPONENTS_AVAILABLE,
            "runtime": RUNTIME_AVAILABLE,
            "deployment": DEPLOYMENT_AVAILABLE,
            "dev_tools": DEV_TOOLS_AVAILABLE,
        },
        "active_processes": len(system_state.active_processes),
        "uptime": str(datetime.now() - system_state.startup_time),
    }
    
    # Add memory info if available
    try:
        import psutil
        diagnostics["system_resources"] = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent if hasattr(psutil, 'disk_usage') else 0
        }
    except ImportError:
        diagnostics["system_resources"] = {"status": "psutil not available"}
    
    return diagnostics

# WebSocket endpoints for real-time updates
try:
    from fastapi import WebSocket, WebSocketDisconnect
    
    @app.websocket("/ws/system-status")
    async def websocket_system_status(websocket: WebSocket):
        """WebSocket endpoint for real-time system status updates."""
        await websocket.accept()
        try:
            while True:
                status_data = {
                    "timestamp": datetime.now().isoformat(),
                    "components": {
                        "network": system_state.network is not None,
                        "runtime": system_state.runtime is not None,
                        "active_processes": len(system_state.active_processes)
                    },
                    "metrics": system_state.metrics.dict()
                }
                
                await websocket.send_json(status_data)
                await asyncio.sleep(5)  # Update every 5 seconds
                
        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            await websocket.close()

except ImportError:
    logger.warning("WebSocket support not available")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={"message": "Endpoint not found", "path": str(request.url.path)}
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error", "detail": str(exc)}
    )

# Frontend serving
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the main dashboard."""
    dashboard_file = Path("frontend/dashboard.html")
    if dashboard_file.exists():
        return dashboard_file.read_text()
    
    # Fallback dashboard
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Enhanced CSP System Dashboard</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
            .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .status-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
            .status-online { background-color: #10b981; }
            .status-offline { background-color: #ef4444; }
            .status-partial { background-color: #f59e0b; }
            h1, h2 { margin-top: 0; }
            .metric { margin: 10px 0; }
            .metric-label { font-weight: 500; }
            .metric-value { font-size: 1.2em; color: #6366f1; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 Enhanced CSP System Dashboard</h1>
                <p>Advanced Computing Systems Platform with AI, Quantum, and Distributed Computing</p>
            </div>
            
            <div class="grid">
                <div class="card">
                    <h2>System Status</h2>
                    <div class="metric">
                        <span class="status-indicator status-online"></span>
                        <span class="metric-label">Core System:</span>
                        <span class="metric-value">Online</span>
                    </div>
                    <div class="metric">
                        <span class="status-indicator status-partial"></span>
                        <span class="metric-label">Components:</span>
                        <span class="metric-value">Partial</span>
                    </div>
                </div>
                
                <div class="card">
                    <h2>Components</h2>
                    <div class="metric">
                        <span class="metric-label">CSP Core:</span>
                        <span class="metric-value" id="csp-core-status">Loading...</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">AI Components:</span>
                        <span class="metric-value" id="ai-status">Loading...</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Runtime:</span>
                        <span class="metric-value" id="runtime-status">Loading...</span>
                    </div>
                </div>
                
                <div class="card">
                    <h2>Quick Actions</h2>
                    <button onclick="runDiagnostics()" style="margin: 5px; padding: 10px 15px; background: #6366f1; color: white; border: none; border-radius: 4px; cursor: pointer;">Run Diagnostics</button>
                    <button onclick="viewLogs()" style="margin: 5px; padding: 10px 15px; background: #10b981; color: white; border: none; border-radius: 4px; cursor: pointer;">View Logs</button>
                    <button onclick="window.open('/docs', '_blank')" style="margin: 5px; padding: 10px 15px; background: #f59e0b; color: white; border: none; border-radius: 4px; cursor: pointer;">API Docs</button>
                </div>
                
                <div class="card">
                    <h2>System Metrics</h2>
                    <div class="metric">
                        <span class="metric-label">Uptime:</span>
                        <span class="metric-value" id="uptime">Loading...</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Active Processes:</span>
                        <span class="metric-value" id="processes">Loading...</span>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            async function loadSystemInfo() {
                try {
                    const response = await fetch('/api/system/info');
                    const data = await response.json();
                    
                    document.getElementById('csp-core-status').textContent = data.components.csp_core_available ? 'Available' : 'Unavailable';
                    document.getElementById('ai-status').textContent = data.components.ai_components_available ? 'Available' : 'Unavailable';
                    document.getElementById('runtime-status').textContent = data.components.runtime_available ? 'Available' : 'Unavailable';
                    document.getElementById('uptime').textContent = data.uptime;
                    document.getElementById('processes').textContent = data.active_processes;
                } catch (error) {
                    console.error('Failed to load system info:', error);
                }
            }
            
            async function runDiagnostics() {
                try {
                    const response = await fetch('/api/test/run-diagnostics', { method: 'POST' });
                    const data = await response.json();
                    alert('Diagnostics completed. Check console for details.');
                    console.log('Diagnostics:', data);
                } catch (error) {
                    alert('Diagnostics failed: ' + error.message);
                }
            }
            
            function viewLogs() {
                alert('Log viewing feature coming soon!');
            }
            
            // Load initial data
            loadSystemInfo();
            
            // Refresh every 30 seconds
            setInterval(loadSystemInfo, 30000);
        </script>
    </body>
    </html>
    """)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
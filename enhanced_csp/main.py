#!/usr/bin/env python3
"""
Enhanced CSP System - Main Application
======================================

Revolutionary AI-to-AI Communication Platform using Communicating Sequential Processes (CSP)
with quantum-inspired protocols, consciousness integration, and emergent behavior detection.

Features:
- FastAPI web server with WebSocket support
- Advanced CSP engine with quantum communication
- AI-powered protocol synthesis
- Real-time monitoring and metrics
- Distributed agent coordination
- Self-healing and optimization
- Visual development tools
- Production deployment support
"""

# ============================================================================
# ENVIRONMENT SETUP (Must be first!)
# ============================================================================
from dotenv import load_dotenv
load_dotenv()  # Load environment variables early

# ============================================================================
# STANDARD LIBRARY IMPORTS
# ============================================================================
import asyncio
import gc
import glob
import json
import logging
import os
import random
import sqlite3
import sys
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timedelta
from enum import Enum
from functools import lru_cache, wraps
from hashlib import sha256
from pathlib import Path
from secrets import token_urlsafe
from traceback import format_exc
from typing import Any, Dict, List, Optional, Union
from weakref import WeakSet

# ============================================================================
# THIRD-PARTY IMPORTS
# ============================================================================
import uvicorn
import yaml
from packaging import version
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
from typing_extensions import Annotated

# FastAPI Framework
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    WebSocket
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Database
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# ============================================================================
# OPTIONAL DEPENDENCIES (with graceful fallbacks)
# ============================================================================
from utils.optional_imports import optional_import

# Network and async
redis_module, REDIS_AVAILABLE = optional_import('redis.asyncio')
aiofiles, AIOFILES_AVAILABLE = optional_import('aiofiles')
aiohttp, AIOHTTP_AVAILABLE = optional_import('aiohttp')
uvloop, UVLOOP_AVAILABLE = optional_import('uvloop')

# Monitoring
prometheus_client, PROMETHEUS_AVAILABLE = optional_import('prometheus_client')
if PROMETHEUS_AVAILABLE:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
else:
    Counter = Histogram = Gauge = generate_latest = None
    CONTENT_TYPE_LATEST = "text/plain"

# System monitoring
psutil, PSUTIL_AVAILABLE = optional_import('psutil')

# ============================================================================
# VALIDATE CRITICAL DEPENDENCIES
# ============================================================================
try:
    import fastapi
    if version.parse(fastapi.__version__) < version.parse("0.100.0"):
        raise ImportError(f"FastAPI {fastapi.__version__} too old, need >= 0.100.0")
except ImportError as e:
    sys.exit(f"Critical dependency error: {e}")

# ============================================================================
# LOCAL APPLICATION IMPORTS
# ============================================================================
from core.config import CSPConfig, load_config
from core.system_state import SystemState
from utils.logging_setup import setup_logging
from utils.lazy_imports import LazyImporter

# Lazy loading for heavy modules
ai_modules = LazyImporter(['ai.engines', 'ai.coordination'])
monitoring_modules = LazyImporter(['monitoring.advanced', 'monitoring.metrics'])
deployment_modules = LazyImporter(['deployment.orchestrator', 'deployment.containers'])

# CSP Components (with fallbacks)
try:
    from core.advanced_csp_core import (
        AdvancedCSPEngine,
        Process,
        AtomicProcess,
        CompositeProcess,
        CompositionOperator,
        ChannelType,
        Event,
        ProcessSignature,
        ProcessContext,
        Channel,
        ProcessMatcher,
        ProtocolEvolution
    )
    CSP_CORE_AVAILABLE = True
    logging.info("✅ CSP Core components loaded")
except ImportError as e:
    logging.warning(f"CSP Core components not available: {e}")
    from core.fallbacks import (
        AdvancedCSPEngine,
        Process,
        AtomicProcess,
        CompositeProcess,
        ProcessSignature
    )
    CSP_CORE_AVAILABLE = False

# AI Components (lazy loaded)
AI_EXTENSIONS_AVAILABLE = False
try:
    # Test if AI modules are available
    from ai.engines import AdvancedCSPEngineWithAI
    AI_EXTENSIONS_AVAILABLE = True
    logging.info("✅ AI extensions available")
except ImportError as e:
    logging.warning(f"AI extensions not available: {e}")

# Runtime Components (lazy loaded)
RUNTIME_AVAILABLE = False
try:
    from runtime.orchestrator import CSPRuntimeOrchestrator
    RUNTIME_AVAILABLE = True
    logging.info("✅ Runtime orchestrator available")
except ImportError as e:
    logging.warning(f"Runtime orchestrator not available: {e}")

# Deployment Components (lazy loaded)
DEPLOYMENT_AVAILABLE = False
try:
    from deployment.orchestrator import CSPDeploymentOrchestrator
    DEPLOYMENT_AVAILABLE = True
    logging.info("✅ Deployment orchestrator available")
except ImportError as e:
    logging.warning(f"Deployment orchestrator not available: {e}")

# Development Tools (lazy loaded)
DEV_TOOLS_AVAILABLE = False
try:
    from dev_tools.visual_designer import CSPDevelopmentTools
    DEV_TOOLS_AVAILABLE = True
    logging.info("✅ Development tools available")
except ImportError as e:
    logging.warning(f"Development tools not available: {e}")

# Monitoring Components (lazy loaded)
MONITORING_AVAILABLE = False
try:
    from monitoring.csp_monitor import CSPMonitor
    MONITORING_AVAILABLE = True
    logging.info("✅ Monitoring components available")
except ImportError as e:
    logging.warning(f"Monitoring components not available: {e}")

# ============================================================================
# CONFIGURATION AND LOGGING SETUP
# ============================================================================
# Setup logging first
setup_logging()
logger = logging.getLogger(__name__)

# Load configuration
config = load_config()

# Validate configuration
if not config.secret_key or config.secret_key == 'dev-secret-key':
    if config.environment == 'production':
        sys.exit("❌ Production environment requires a secure secret key!")
    logger.warning("⚠️ Using default secret key in development mode")

# ============================================================================
# PROMETHEUS METRICS SETUP
# ============================================================================
if PROMETHEUS_AVAILABLE:
    # Request metrics
    REQUEST_COUNT = Counter(
        'csp_requests_total', 
        'Total requests', 
        ['method', 'endpoint', 'status']
    )
    REQUEST_DURATION = Histogram(
        'csp_request_duration_seconds', 
        'Request duration'
    )
    
    # System metrics
    ACTIVE_PROCESSES = Gauge('csp_active_processes', 'Number of active CSP processes')
    WEBSOCKET_CONNECTIONS = Gauge('csp_websocket_connections', 'Number of WebSocket connections')
    SYSTEM_HEALTH = Gauge('csp_system_health', 'System health status (0-1)')
    
    # Custom middleware for metrics
    class MetricsMiddleware:
        def __init__(self, app):
            self.app = app
        
        async def __call__(self, scope, receive, send):
            if scope["type"] == "http":
                start_time = time.time()
                
                async def send_with_metrics(message):
                    if message["type"] == "http.response.start":
                        REQUEST_COUNT.labels(
                            method=scope["method"],
                            endpoint=scope["path"],
                            status=message["status"]
                        ).inc()
                        REQUEST_DURATION.observe(time.time() - start_time)
                    await send(message)
                
                await self.app(scope, receive, send_with_metrics)
            else:
                await self.app(scope, receive, send)
else:
    # Fallback metrics
    REQUEST_COUNT = REQUEST_DURATION = None
    ACTIVE_PROCESSES = WEBSOCKET_CONNECTIONS = SYSTEM_HEALTH = None
    
    class MetricsMiddleware:
        def __init__(self, app):
            self.app = app
        async def __call__(self, scope, receive, send):
            await self.app(scope, receive, send)

# ============================================================================
# GLOBAL SYSTEM STATE
# ============================================================================
system_state = SystemState()

# Infrastructure mock data for frontend
infrastructure_services = [
    {"name": "Web Server", "status": "running", "uptime": "1d 0h", "port": 8000},
    {"name": "Database", "status": "running", "uptime": "1d 0h", "port": 5432},
    {"name": "Redis Cache", "status": "running", "uptime": "1d 0h", "port": 6379},
]
infrastructure_alerts: List[Dict[str, Any]] = []
infrastructure_backups: List[Dict[str, Any]] = []
maintenance_mode = False

def generate_infrastructure_metrics() -> Dict[str, Any]:
    """Generate sample infrastructure metrics."""
    return {
        "cpu": {"current": psutil.cpu_percent() if PSUTIL_AVAILABLE else random.uniform(20, 80), "max": 100, "unit": "%"},
        "memory": {"current": psutil.virtual_memory().percent if PSUTIL_AVAILABLE else random.uniform(20, 80), "max": 100, "unit": "%"},
        "disk": {"current": psutil.disk_usage('/').percent if PSUTIL_AVAILABLE else random.uniform(20, 80), "max": 100, "unit": "%"},
        "network": {"current": random.uniform(0, 100), "max": 100, "unit": "%"},
        "uptime": {"current": 99.9, "max": 100, "unit": "%"},
        "requests": {"current": random.randint(500, 1500), "max": None, "unit": "/min"},
    }

def get_infrastructure_status() -> Dict[str, Any]:
    """Return current infrastructure status."""
    return {
        "services": infrastructure_services,
        "alerts": infrastructure_alerts,
        "backups": infrastructure_backups,
        "maintenance_mode": maintenance_mode,
        "metrics": generate_infrastructure_metrics()
    }

# ============================================================================
# DATABASE SETUP
# ============================================================================
async def setup_database():
    """Initialize database connection"""
    try:
        if config.database.url.startswith('sqlite'):
            system_state.db_engine = create_async_engine(
                config.database.url,
                echo=config.debug,
                pool_pre_ping=True
            )
        else:
            system_state.db_engine = create_async_engine(
                config.database.url,
                echo=config.debug,
                pool_size=config.database.pool_size,
                max_overflow=config.database.max_overflow,
                pool_pre_ping=True
            )
        
        # Test connection
        async with system_state.db_engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        
        logger.info("✅ Database connection established")
    except Exception as e:
        logger.error(f"❌ Database setup failed: {e}")
        raise

async def setup_redis():
    """Initialize Redis connection"""
    if not REDIS_AVAILABLE:
        logger.warning("Redis not available, skipping Redis setup")
        return
    
    try:
        system_state.redis_client = redis_module.Redis(
            host=config.redis.host,
            port=config.redis.port,
            password=config.redis.password,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5
        )
        
        # Test connection
        await system_state.redis_client.ping()
        logger.info("✅ Redis connection established")
    except Exception as e:
        logger.error(f"❌ Redis setup failed: {e}")
        system_state.redis_client = None

async def initialize_csp_system():
    """Initialize CSP components"""
    try:
        # Initialize CSP Engine
        if CSP_CORE_AVAILABLE:
            system_state.csp_engine = AdvancedCSPEngine()
            logger.info("✅ CSP Engine initialized")
        
        # Initialize AI Engine (lazy loaded)
        if AI_EXTENSIONS_AVAILABLE:
            ai_engine_class = ai_modules.get_class('ai.engines', 'AdvancedCSPEngineWithAI')
            if ai_engine_class:
                system_state.ai_engine = ai_engine_class()
                logger.info("✅ AI Engine initialized")
        
        # Initialize Runtime Orchestrator (lazy loaded)
        if RUNTIME_AVAILABLE:
            system_state.runtime_orchestrator = CSPRuntimeOrchestrator()
            await system_state.runtime_orchestrator.start()
            logger.info("✅ Runtime Orchestrator initialized")
        
        # Initialize Deployment Orchestrator (lazy loaded)
        if DEPLOYMENT_AVAILABLE:
            deployment_class = deployment_modules.get_class('deployment.orchestrator', 'CSPDeploymentOrchestrator')
            if deployment_class:
                system_state.deployment_orchestrator = deployment_class()
                logger.info("✅ Deployment Orchestrator initialized")
        
        # Initialize Development Tools (lazy loaded)
        if DEV_TOOLS_AVAILABLE:
            system_state.dev_tools = CSPDevelopmentTools()
            logger.info("✅ Development Tools initialized")
        
        # Initialize Monitor (lazy loaded)
        if MONITORING_AVAILABLE:
            monitor_class = monitoring_modules.get_class('monitoring.csp_monitor', 'CSPMonitor')
            if monitor_class:
                system_state.monitor = monitor_class()
                await system_state.monitor.start()
                logger.info("✅ Monitor initialized")
        
    except Exception as e:
        logger.error(f"❌ CSP system initialization failed: {e}")
        raise

# ============================================================================
# APPLICATION LIFESPAN MANAGEMENT
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting Enhanced CSP System...")
    
    try:
        await setup_database()
        await setup_redis()
        await initialize_csp_system()
        
        # Update system health
        if SYSTEM_HEALTH:
            SYSTEM_HEALTH.set(1.0)
        logger.info("✅ System startup completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        if SYSTEM_HEALTH:
            SYSTEM_HEALTH.set(0.5)  # Partial functionality
        # Don't raise - allow server to start with reduced functionality
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Enhanced CSP System...")
    
    try:
        # Close WebSocket connections
        for ws in list(system_state.active_websockets):
            try:
                await ws.close()
            except:
                pass
        
        # Shutdown components
        if system_state.runtime_orchestrator:
            await system_state.runtime_orchestrator.stop()
        
        if system_state.monitor:
            await system_state.monitor.stop()
        
        if system_state.redis_client:
            await system_state.redis_client.close()
        
        if system_state.db_engine:
            await system_state.db_engine.dispose()
        
        logger.info("✅ System shutdown completed")
        
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}")

# ============================================================================
# FASTAPI APPLICATION SETUP
# ============================================================================
app = FastAPI(
    title=config.app_name,
    description="Revolutionary AI-to-AI Communication Platform using CSP",
    version=config.version,
    debug=config.debug,
    lifespan=lifespan
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
if PROMETHEUS_AVAILABLE:
    app.add_middleware(MetricsMiddleware)

# ============================================================================
# SECURITY AND AUTHENTICATION
# ============================================================================
security = HTTPBearer(auto_error=False)

@lru_cache(maxsize=1000)
def verify_api_key(api_key: str) -> bool:
    """Verify API key with caching"""
    # Use secure comparison to prevent timing attacks
    expected = config.secret_key.encode()
    provided = api_key.encode()
    return len(expected) == len(provided) and sha256(expected).digest() == sha256(provided).digest()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Enhanced authentication dependency"""
    if not config.enable_auth:
        return {"user": "anonymous", "authenticated": False}
    
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # Verify API key securely
    if verify_api_key(credentials.credentials):
        return {"user": "api_user", "authenticated": True}
    
    raise HTTPException(status_code=401, detail="Invalid authentication")

# ============================================================================
# DATABASE SESSION DEPENDENCY
# ============================================================================
async def get_db_session() -> AsyncSession:
    """Database session dependency"""
    if not system_state.db_engine:
        raise HTTPException(status_code=503, detail="Database not available")
    
    async with AsyncSession(system_state.db_engine) as session:
        try:
            yield session
        finally:
            await session.close()

# ============================================================================
# CORE API ENDPOINTS
# ============================================================================
@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "app": config.app_name,
        "version": config.version,
        "status": "running",
        "uptime": str(datetime.now() - system_state.startup_time),
        "features": {
            "csp_engine": system_state.csp_engine is not None,
            "ai_integration": system_state.ai_engine is not None,
            "runtime_orchestrator": system_state.runtime_orchestrator is not None,
            "deployment": system_state.deployment_orchestrator is not None,
            "development_tools": system_state.dev_tools is not None,
            "monitoring": system_state.monitor is not None
        },
        "components": {
            "csp_core": CSP_CORE_AVAILABLE,
            "ai_extensions": AI_EXTENSIONS_AVAILABLE,
            "runtime": RUNTIME_AVAILABLE,
            "deployment": DEPLOYMENT_AVAILABLE,
            "dev_tools": DEV_TOOLS_AVAILABLE,
            "monitoring": MONITORING_AVAILABLE,
            "redis": REDIS_AVAILABLE,
            "prometheus": PROMETHEUS_AVAILABLE,
            "psutil": PSUTIL_AVAILABLE
        }
    }

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    health_data = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system": {
            "cpu_percent": psutil.cpu_percent() if PSUTIL_AVAILABLE else 0.0,
            "memory_percent": psutil.virtual_memory().percent if PSUTIL_AVAILABLE else 0.0,
            "disk_percent": psutil.disk_usage('/').percent if PSUTIL_AVAILABLE else 0.0
        },
        "components": {
            "database": system_state.db_engine is not None,
            "redis": system_state.redis_client is not None,
            "csp_engine": system_state.csp_engine is not None,
            "ai_engine": system_state.ai_engine is not None
        },
        "active_processes": len(system_state.active_processes),
        "websocket_connections": len(system_state.active_websockets)
    }
    
    # Update metrics
    if ACTIVE_PROCESSES:
        ACTIVE_PROCESSES.set(len(system_state.active_processes))
    if WEBSOCKET_CONNECTIONS:
        WEBSOCKET_CONNECTIONS.set(len(system_state.active_websockets))
    
    return health_data

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    if generate_latest:
        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    return PlainTextResponse("metrics unavailable", media_type=CONTENT_TYPE_LATEST)

# ============================================================================
# CSP PROCESS MANAGEMENT API
# ============================================================================
@app.post("/api/processes")
async def create_process(
    process_data: dict,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session)
):
    """Create a new CSP process"""
    try:
        if not system_state.csp_engine:
            raise HTTPException(status_code=503, detail="CSP engine not available")
        
        # Validate process limit
        if len(system_state.active_processes) >= config.runtime.max_processes:
            raise HTTPException(status_code=429, detail="Process limit exceeded")
        
        # Create process based on type
        process_type = process_data.get("type", "atomic")
        process_id = str(uuid.uuid4())
        
        if process_type == "atomic":
            process = AtomicProcess(
                name=process_data.get("name", f"process_{process_id}"),
                signature=ProcessSignature(
                    inputs=process_data.get("inputs", []),
                    outputs=process_data.get("outputs", [])
                )
            )
        else:
            process = CompositeProcess(
                name=process_data.get("name", f"composite_{process_id}"),
                processes=[]
            )
        
        # Register process
        await system_state.csp_engine.start_process(process)
        system_state.active_processes[process_id] = process
        
        # Update metrics
        if ACTIVE_PROCESSES:
            ACTIVE_PROCESSES.set(len(system_state.active_processes))
        
        return {
            "process_id": process_id,
            "name": process.name,
            "type": process_type,
            "status": "created"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Process creation failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/processes")
async def list_processes(user: dict = Depends(get_current_user)):
    """List all active processes"""
    processes = []
    for process_id, process in system_state.active_processes.items():
        processes.append({
            "process_id": process_id,
            "name": process.name,
            "type": type(process).__name__,
            "status": getattr(process, 'state', 'unknown')
        })
    
    return {"processes": processes, "count": len(processes)}

@app.delete("/api/processes/{process_id}")
async def stop_process(
    process_id: str,
    user: dict = Depends(get_current_user),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Stop a specific process"""
    if process_id not in system_state.active_processes:
        raise HTTPException(status_code=404, detail="Process not found")
    
    try:
        process = system_state.active_processes[process_id]
        
        # Stop process in background
        background_tasks.add_task(cleanup_process, process_id, process)
        
        return {"process_id": process_id, "status": "stopping"}
    except Exception as e:
        logger.error(f"Process stop failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

async def cleanup_process(process_id: str, process):
    """Background task to cleanup process resources"""
    try:
        await system_state.csp_engine.stop_process(process)
        del system_state.active_processes[process_id]
        
        # Update metrics
        if ACTIVE_PROCESSES:
            ACTIVE_PROCESSES.set(len(system_state.active_processes))
        
        logger.info(f"Process {process_id} cleaned up successfully")
    except Exception as e:
        logger.error(f"Process cleanup failed: {e}")

# ============================================================================
# AI COLLABORATION API
# ============================================================================
@app.post("/api/ai/collaborate")
async def create_ai_collaboration(
    collaboration_data: dict,
    user: dict = Depends(get_current_user)
):
    """Create an AI collaboration session"""
    if not system_state.ai_engine:
        raise HTTPException(status_code=503, detail="AI engine not available")
    
    try:
        # Lazy load AI components
        ai_agent_class = ai_modules.get_class('ai.engines', 'AIAgent')
        llm_capability_class = ai_modules.get_class('ai.engines', 'LLMCapability')
        coordinator_class = ai_modules.get_class('ai.coordination', 'MultiAgentReasoningCoordinator')
        
        if not all([ai_agent_class, llm_capability_class, coordinator_class]):
            raise HTTPException(status_code=503, detail="AI components not available")
        
        # Create AI agents
        agents = []
        for agent_spec in collaboration_data.get("agents", []):
            capability = llm_capability_class(
                model_name=agent_spec.get("model", config.ai.default_model),
                specialized_domain=agent_spec.get("domain")
            )
            agent = ai_agent_class(
                name=agent_spec.get("name"),
                capabilities=[capability]
            )
            agents.append(agent)
        
        # Create collaborative process
        coordinator = coordinator_class(agents)
        collaboration_id = str(uuid.uuid4())
        
        # Store collaboration
        system_state.active_processes[collaboration_id] = coordinator
        
        return {
            "collaboration_id": collaboration_id,
            "agents": [{"name": a.name, "capabilities": len(a.capabilities)} for a in agents],
            "status": "created"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"AI collaboration creation failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# ============================================================================
# WEBSOCKET ENDPOINTS
# ============================================================================
class ConnectionManager:
    """WebSocket connection manager with limits"""
    
    def __init__(self, max_connections: int = 1000):
        self.connections: WeakSet[WebSocket] = WeakSet()
        self.max_connections = max_connections
    
    async def connect(self, websocket: WebSocket) -> bool:
        """Connect websocket with connection limit"""
        if len(self.connections) >= self.max_connections:
            return False
        
        await websocket.accept()
        self.connections.add(websocket)
        system_state.active_websockets.append(websocket)
        return True
    
    def disconnect(self, websocket: WebSocket):
        """Disconnect websocket"""
        with suppress(ValueError):
            system_state.active_websockets.remove(websocket)

connection_manager = ConnectionManager()

@app.websocket("/ws/system")
async def websocket_system_monitor(websocket: WebSocket):
    """WebSocket for real-time system monitoring"""
    if not await connection_manager.connect(websocket):
        await websocket.close(code=1008, reason="Connection limit exceeded")
        return
    
    try:
        while True:
            # Send system status
            status = {
                "timestamp": datetime.now().isoformat(),
                "active_processes": len(system_state.active_processes),
                "system_health": {
                    "cpu": psutil.cpu_percent() if PSUTIL_AVAILABLE else 0.0,
                    "memory": psutil.virtual_memory().percent if PSUTIL_AVAILABLE else 0.0,
                    "disk": psutil.disk_usage('/').percent if PSUTIL_AVAILABLE else 0.0
                },
                "websocket_connections": len(system_state.active_websockets),
                "components": {
                    "csp_engine": system_state.csp_engine is not None,
                    "ai_engine": system_state.ai_engine is not None,
                    "database": system_state.db_engine is not None,
                    "redis": system_state.redis_client is not None
                }
            }
            
            await websocket.send_json(status)
            await asyncio.sleep(5)  # Update every 5 seconds
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        connection_manager.disconnect(websocket)

# ============================================================================
# STATIC FILES AND UI
# ============================================================================
def serve_html_page(page_name: str, page_title: str) -> HTMLResponse:
    """Serve HTML page with enhanced error handling"""
    try:
        page_path = Path(f"frontend/pages/{page_name}.html")
        
        if page_path.exists():
            with open(page_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return HTMLResponse(content)
        else:
            # Return enhanced 404 page
            return HTMLResponse(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Page Not Found - {config.app_name}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                    .container {{ max-width: 600px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; }}
                    .error {{ color: #e74c3c; }}
                    .back-link {{ color: #3498db; text-decoration: none; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1 class="error">❌ Page Not Found</h1>
                    <p>The page <strong>{page_name}</strong> could not be found.</p>
                    <p>Expected location: <code>frontend/pages/{page_name}.html</code></p>
                    <p><a href="/" class="back-link">← Back to Home</a></p>
                </div>
            </body>
            </html>
            """, status_code=404)
    except Exception as e:
        logger.error(f"Error serving page {page_name}: {e}")
        return HTMLResponse(f"Internal server error: {e}", status_code=500)

# Enhanced home page
@app.get("/home", response_class=HTMLResponse)
async def enhanced_home():
    """Enhanced home page with system status"""
    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{config.app_name} - Home</title>
        <style>
            body {{ 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 0; padding: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; min-height: 100vh;
            }}
            .container {{ max-width: 1200px; margin: 0 auto; padding: 40px 20px; }}
            .header {{ text-align: center; margin-bottom: 60px; }}
            .header h1 {{ font-size: 3em; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }}
            .header p {{ font-size: 1.2em; opacity: 0.9; margin: 20px 0; }}
            .features {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 30px; }}
            .feature-card {{ 
                background: rgba(255,255,255,0.1); padding: 30px; border-radius: 15px;
                backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.2);
                transition: transform 0.3s ease;
            }}
            .feature-card:hover {{ transform: translateY(-5px); }}
            .feature-card h3 {{ margin-top: 0; font-size: 1.5em; }}
            .nav-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-top: 40px; }}
            .nav-link {{ 
                display: block; padding: 15px 20px; background: rgba(255,255,255,0.2);
                border-radius: 10px; text-decoration: none; color: white; text-align: center;
                transition: all 0.3s ease; border: 1px solid rgba(255,255,255,0.3);
            }}
            .nav-link:hover {{ background: rgba(255,255,255,0.3); transform: scale(1.05); }}
            .status {{ text-align: center; margin: 40px 0; }}
            .status-badge {{ 
                display: inline-block; padding: 8px 16px; border-radius: 20px;
                background: rgba(39, 174, 96, 0.8); margin: 0 10px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 {config.app_name}</h1>
                <p>Revolutionary AI-to-AI Communication Platform</p>
                <div class="status">
                    <span class="status-badge">✅ System Running</span>
                    <span class="status-badge">🔧 Version {config.version}</span>
                    <span class="status-badge">🌍 {config.environment.title()}</span>
                </div>
            </div>
            
            <div class="features">
                <div class="feature-card">
                    <h3>🧠 CSP Engine</h3>
                    <p>Advanced Communicating Sequential Processes with quantum-inspired protocols and emergent behavior detection.</p>
                </div>
                <div class="feature-card">
                    <h3>🤖 AI Integration</h3>
                    <p>Multi-agent coordination with LLM capabilities, consciousness integration, and adaptive learning systems.</p>
                </div>
                <div class="feature-card">
                    <h3>📊 Real-time Monitoring</h3>
                    <p>Comprehensive system monitoring with Prometheus metrics, health checks, and performance analytics.</p>
                </div>
                <div class="feature-card">
                    <h3>🛠️ Development Tools</h3>
                    <p>Visual designers, code generators, and debugging tools for rapid CSP application development.</p>
                </div>
            </div>
            
            <div class="nav-grid">
                <a href="/dashboard" class="nav-link">📊 Dashboard</a>
                <a href="/processes" class="nav-link">⚙️ Processes</a>
                <a href="/ai" class="nav-link">🤖 AI Hub</a>
                <a href="/monitoring" class="nav-link">📈 Monitoring</a>
                <a href="/security" class="nav-link">🔐 Security</a>
                <a href="/docs" class="nav-link">📚 API Docs</a>
                <a href="/health" class="nav-link">❤️ Health Check</a>
                <a href="/metrics" class="nav-link">📊 Metrics</a>
            </div>
        </div>
    </body>
    </html>
    """)

# Mount static files if available
static_dir = Path("frontend/static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Template support
templates_dir = Path("frontend/templates")
if templates_dir.exists():
    templates = Jinja2Templates(directory=str(templates_dir))
    
    @app.get("/ui", response_class=HTMLResponse)
    async def web_ui(request: Request):
        """Web UI for the CSP system"""
        return templates.TemplateResponse("index.html", {"request": request})

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main entry point with enhanced configuration"""
    logger.info(f"🚀 Starting {config.app_name} v{config.version}")
    logger.info(f"Environment: {config.environment}")
    logger.info(f"Host: {config.host}:{config.port}")
    logger.info(f"Debug mode: {config.debug}")
    
    # Set up event loop optimization
    if UVLOOP_AVAILABLE and sys.platform != 'win32':
        try:
            uvloop.install()
            logger.info("✅ uvloop installed for better performance")
        except Exception as e:
            logger.warning(f"uvloop installation failed: {e}")
    else:
        logger.info("Using default asyncio event loop")
    
    # Garbage collection optimization
    gc.set_threshold(700, 10, 10)
    
    try:
        uvicorn.run(
            app,
            host=config.host,
            port=config.port,
            workers=config.workers,
            reload=config.reload and config.debug,
            log_level="info" if not config.debug else "debug",
            access_log=True,
            # Performance optimizations
            loop="uvloop" if UVLOOP_AVAILABLE else "asyncio",
            http="httptools",
            ws="websockets",
            lifespan="on",
            # Security
            server_header=False,
            date_header=False
        )
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
"""
Configuration Management
=======================

Centralized configuration with environment variable support and validation.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from functools import lru_cache
import yaml
import json
import logging
from pydantic import BaseModel, Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class DatabaseConfig(BaseModel):
    """Database configuration."""
    model_config = ConfigDict(extra='forbid')
    
    url: str = Field(default="sqlite+aiosqlite:///./csp_system.db", description="Database URL")
    pool_size: int = Field(default=5, ge=1, le=50, description="Connection pool size")
    max_overflow: int = Field(default=10, ge=0, le=100, description="Max overflow connections")
    pool_timeout: int = Field(default=30, ge=1, description="Pool timeout in seconds")
    pool_recycle: int = Field(default=3600, ge=300, description="Pool recycle time in seconds")
    echo: bool = Field(default=False, description="Echo SQL queries")
    
    @field_validator('url')
    @classmethod
    def validate_database_url(cls, v):
        """Validate database URL format."""
        if not v or not isinstance(v, str):
            raise ValueError("Database URL must be a non-empty string")
        
        valid_schemes = ['sqlite', 'sqlite+aiosqlite', 'postgresql', 'postgresql+asyncpg', 'mysql+aiomysql']
        scheme = v.split('://')[0] if '://' in v else ''
        
        if scheme not in valid_schemes:
            logger.warning(f"Database scheme '{scheme}' may not be supported")
        
        return v


class RedisConfig(BaseModel):
    """Redis configuration."""
    model_config = ConfigDict(extra='forbid')
    
    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, ge=1, le=65535, description="Redis port")
    password: Optional[str] = Field(default=None, description="Redis password")
    db: int = Field(default=0, ge=0, le=15, description="Redis database number")
    max_connections: int = Field(default=20, ge=1, le=1000, description="Max connections")
    socket_timeout: int = Field(default=5, ge=1, description="Socket timeout in seconds")
    socket_connect_timeout: int = Field(default=5, ge=1, description="Connection timeout")
    decode_responses: bool = Field(default=True, description="Decode responses to strings")


class MonitoringConfig(BaseModel):
    """Monitoring and metrics configuration."""
    model_config = ConfigDict(extra='forbid')
    
    enable_prometheus: bool = Field(default=True, description="Enable Prometheus metrics")
    enable_health_checks: bool = Field(default=True, description="Enable health check endpoints")
    metrics_port: int = Field(default=9090, ge=1024, le=65535, description="Metrics server port")
    health_check_interval: int = Field(default=30, ge=5, description="Health check interval in seconds")
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format (json or text)")
    enable_structured_logs: bool = Field(default=True, description="Enable structured logging")
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    @field_validator('log_format')
    @classmethod
    def validate_log_format(cls, v):
        """Validate log format."""
        valid_formats = ['json', 'text', 'structured']
        if v.lower() not in valid_formats:
            raise ValueError(f"Log format must be one of: {valid_formats}")
        return v.lower()


class AIConfig(BaseModel):
    """AI and machine learning configuration."""
    model_config = ConfigDict(extra='forbid')
    
    enable_ai: bool = Field(default=True, description="Enable AI features")
    default_model: str = Field(default="gpt-3.5-turbo", description="Default LLM model")
    max_tokens: int = Field(default=4096, ge=1, le=32768, description="Max tokens per request")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Model temperature")
    timeout: int = Field(default=30, ge=5, le=300, description="AI request timeout in seconds")
    max_concurrent_requests: int = Field(default=10, ge=1, le=100, description="Max concurrent AI requests")
    enable_caching: bool = Field(default=True, description="Enable response caching")
    cache_ttl: int = Field(default=3600, ge=60, description="Cache TTL in seconds")


class SecurityConfig(BaseModel):
    """Security configuration."""
    model_config = ConfigDict(extra='forbid')
    
    enable_auth: bool = Field(default=True, description="Enable authentication")
    secret_key: str = Field(description="Secret key for signing")
    api_key_header: str = Field(default="X-CSP-API-Key", description="API key header name")
    jwt_algorithm: str = Field(default="HS256", description="JWT signing algorithm")
    jwt_expiration: int = Field(default=3600, ge=300, description="JWT expiration in seconds")
    enable_cors: bool = Field(default=True, description="Enable CORS")
    cors_origins: List[str] = Field(default=["*"], description="Allowed CORS origins")
    enable_rate_limiting: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_requests: int = Field(default=100, ge=1, description="Requests per minute")
    enable_https_only: bool = Field(default=False, description="Force HTTPS only")
    
    @field_validator('secret_key')
    @classmethod
    def validate_secret_key(cls, v):
        """Validate secret key strength."""
        if not v:
            raise ValueError("Secret key cannot be empty")
        
        if len(v) < 32:
            logger.warning("Secret key should be at least 32 characters long")
        
        if v == 'dev-secret-key':
            logger.warning("Using default development secret key")
        
        return v
    
    @field_validator('jwt_algorithm')
    @classmethod
    def validate_jwt_algorithm(cls, v):
        """Validate JWT algorithm."""
        valid_algorithms = ['HS256', 'HS384', 'HS512', 'RS256', 'RS384', 'RS512']
        if v not in valid_algorithms:
            raise ValueError(f"JWT algorithm must be one of: {valid_algorithms}")
        return v


class RuntimeSettings(BaseModel):
    """Runtime configuration."""
    model_config = ConfigDict(extra='forbid')
    
    max_processes: int = Field(default=1000, ge=1, le=10000, description="Max CSP processes")
    max_websockets: int = Field(default=1000, ge=1, le=10000, description="Max WebSocket connections")
    process_timeout: int = Field(default=300, ge=30, description="Process timeout in seconds")
    websocket_timeout: int = Field(default=300, ge=30, description="WebSocket timeout in seconds")
    background_task_limit: int = Field(default=100, ge=1, description="Max background tasks")
    enable_gc_optimization: bool = Field(default=True, description="Enable garbage collection optimization")
    gc_threshold: tuple = Field(default=(700, 10, 10), description="GC threshold values")
    enable_uvloop: bool = Field(default=True, description="Enable uvloop if available")


class CSPConfig(BaseSettings):
    """Main CSP system configuration."""
    model_config = ConfigDict(
        env_prefix='CSP_',
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'
    )
    
    # Core application settings
    app_name: str = Field(default="Enhanced CSP System", description="Application name")
    version: str = Field(default="2.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    environment: str = Field(default="development", description="Environment name")
    
    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, ge=1, le=65535, description="Server port")
    workers: int = Field(default=1, ge=1, le=32, description="Number of workers")
    reload: bool = Field(default=False, description="Enable auto-reload")
    
    # Component configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    security: SecurityConfig = Field(default_factory=lambda: SecurityConfig(secret_key=os.getenv('CSP_SECRET_KEY', 'dev-secret-key')))
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    
    # Additional settings
    config_file: Optional[str] = Field(default=None, description="Configuration file path")
    data_dir: str = Field(default="./data", description="Data directory")
    log_dir: str = Field(default="./logs", description="Log directory")
    temp_dir: str = Field(default="./temp", description="Temporary directory")
    
    @field_validator('port')
    @classmethod
    def validate_port(cls, v):
        """Validate port number."""
        if not 1 <= v <= 65535:
            raise ValueError('Port must be between 1 and 65535')
        return v
    
    @field_validator('environment')
    @classmethod
    def validate_environment(cls, v):
        """Validate environment name."""
        valid_environments = ['development', 'testing', 'staging', 'production']
        if v.lower() not in valid_environments:
            logger.warning(f"Environment '{v}' is not standard. Expected: {valid_environments}")
        return v.lower()
    
    @field_validator('workers')
    @classmethod
    def validate_workers(cls, v):
        """Validate worker count."""
        import multiprocessing
        max_workers = multiprocessing.cpu_count() * 2
        if v > max_workers:
            logger.warning(f"Worker count {v} exceeds recommended maximum {max_workers}")
        return v
    
    def model_post_init(self, __context):
        """Post-initialization validation."""
        # Create directories
        for dir_path in [self.data_dir, self.log_dir, self.temp_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Validate production settings
        if self.environment == 'production':
            if self.debug:
                logger.warning("Debug mode should be disabled in production")
            
            if self.security.secret_key == 'dev-secret-key':
                raise ValueError("Production environment requires a secure secret key")
            
            if not self.security.enable_auth:
                logger.warning("Authentication should be enabled in production")
    
    def get_database_url(self) -> str:
        """Get the database URL with environment variable substitution."""
        url = self.database.url
        
        # Substitute environment variables in database URL
        if '${' in url and '}' in url:
            import re
            def replace_env_var(match):
                var_name = match.group(1)
                return os.getenv(var_name, match.group(0))
            
            url = re.sub(r'\$\{([^}]+)\}', replace_env_var, url)
        
        return url
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()
    
    def to_json(self) -> str:
        """Convert configuration to JSON string."""
        return self.model_dump_json(indent=2)
    
    def save_to_file(self, file_path: Union[str, Path]):
        """Save configuration to file."""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'w') as f:
                f.write(self.to_json())
        elif file_path.suffix.lower() in ['.yml', '.yaml']:
            with open(file_path, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
        else:
            raise ValueError("Unsupported file format. Use .json, .yml, or .yaml")
        
        logger.info(f"Configuration saved to {file_path}")


@lru_cache(maxsize=1)
def load_config() -> CSPConfig:
    """
    Load configuration from multiple sources with caching.
    
    Priority order:
    1. Environment variables
    2. Configuration file (if specified)
    3. Default values
    """
    config_file = os.getenv('CSP_CONFIG_FILE') or os.getenv('CSP_CONFIG_PATH')
    config_data = {}
    
    # Load from configuration file if specified
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                if config_file.endswith(('.yml', '.yaml')):
                    config_data = yaml.safe_load(f) or {}
                elif config_file.endswith('.json'):
                    config_data = json.load(f) or {}
                else:
                    logger.warning(f"Unsupported config file format: {config_file}")
            
            logger.info(f"Configuration loaded from: {config_file}")
        except Exception as e:
            logger.error(f"Failed to load config file {config_file}: {e}")
            sys.exit(1)
    else:
        if config_file:
            logger.warning(f"Config file not found: {config_file}")
        logger.info("Using environment variables and defaults")
    
    # Create configuration instance
    try:
        # Merge file config with environment variables
        if config_data:
            # Environment variables take precedence
            config = CSPConfig(**config_data)
        else:
            config = CSPConfig()
        
        logger.info(f"Configuration loaded successfully")
        logger.debug(f"Environment: {config.environment}")
        logger.debug(f"Debug mode: {config.debug}")
        
        return config
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        sys.exit(1)


def reload_config() -> CSPConfig:
    """Reload configuration (clears cache)."""
    load_config.cache_clear()
    return load_config()


def validate_config_file(file_path: Union[str, Path]) -> bool:
    """Validate a configuration file without loading it."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.error(f"Configuration file does not exist: {file_path}")
        return False
    
    try:
        with open(file_path, 'r') as f:
            if file_path.suffix.lower() in ['.yml', '.yaml']:
                config_data = yaml.safe_load(f) or {}
            elif file_path.suffix.lower() == '.json':
                config_data = json.load(f) or {}
            else:
                logger.error(f"Unsupported file format: {file_path}")
                return False
        
        # Validate by creating config instance
        CSPConfig(**config_data)
        logger.info(f"Configuration file is valid: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Configuration file validation failed: {e}")
        return False


# Example usage and testing
if __name__ == "__main__":
    import tempfile
    
    # Test configuration loading
    print("Testing configuration...")
    
    # Test default configuration
    config = load_config()
    print(f"Default config: {config.app_name} v{config.version}")
    print(f"Environment: {config.environment}")
    print(f"Database URL: {config.get_database_url()}")
    
    # Test configuration serialization
    print(f"\nConfiguration as JSON:")
    print(config.to_json())
    
    # Test saving to file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config.save_to_file(f.name)
        print(f"Configuration saved to: {f.name}")
        
        # Test validation
        is_valid = validate_config_file(f.name)
        print(f"Configuration file is valid: {is_valid}")
    
    # Test environment variable override
    os.environ['CSP_APP_NAME'] = 'Test CSP System'
    os.environ['CSP_DEBUG'] = 'true'
    
    config_with_env = reload_config()
    print(f"\nWith environment overrides:")
    print(f"App name: {config_with_env.app_name}")
    print(f"Debug: {config_with_env.debug}")
    
    print("✅ Configuration test completed")
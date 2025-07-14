# enhanced_csp/network/ai_controller.py
"""
AI Network Controller - Autonomous Network Management
====================================================
Fine-tuned small AI model to run and optimize the entire network automatically
"""

import asyncio
import logging
import json
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor

# Import local components
from ..core.config import NetworkConfig
from ..core.types import NodeID, NetworkMessage, MessageType
from ..ml_routing import MLRoutePredictor, NetworkDataCollector, NetworkMetrics
from ..utils import get_logger

logger = get_logger(__name__)

# ============================================================================
# AI MODEL BACKENDS
# ============================================================================

class ModelBackend(Enum):
    OLLAMA = "ollama"
    LLAMACPP = "llamacpp"
    TRANSFORMERS = "transformers"
    ONNX = "onnx"
    CUSTOM = "custom"

@dataclass
class AIModelConfig:
    """Configuration for the AI network controller"""
    # Model configuration
    backend: ModelBackend = ModelBackend.OLLAMA
    model_name: str = "phi3:mini"  # Small, efficient model
    model_path: Optional[str] = None
    api_url: str = "http://localhost:11434"
    
    # Fine-tuning parameters
    learning_rate: float = 1e-4
    batch_size: int = 32
    max_sequence_length: int = 512
    fine_tune_epochs: int = 10
    
    # Network decision parameters
    decision_interval: float = 5.0  # seconds
    prediction_window: int = 60  # seconds
    max_concurrent_decisions: int = 10
    
    # Training data collection
    collect_training_data: bool = True
    training_data_file: str = "network_training_data.jsonl"
    max_training_samples: int = 100000

class NetworkDecision:
    """Represents a network management decision made by AI"""
    
    def __init__(self, decision_type: str, parameters: Dict[str, Any], 
                 confidence: float, reasoning: str):
        self.decision_type = decision_type
        self.parameters = parameters
        self.confidence = confidence
        self.reasoning = reasoning
        self.timestamp = time.time()
        self.executed = False
        self.result = None

class NetworkState:
    """Complete network state representation for AI model"""
    
    def __init__(self):
        self.timestamp = time.time()
        self.nodes = {}
        self.connections = {}
        self.routing_table = {}
        self.traffic_patterns = {}
        self.performance_metrics = {}
        self.security_alerts = []
        self.resource_usage = {}
        
    def to_prompt(self) -> str:
        """Convert network state to text prompt for AI model"""
        return f"""
NETWORK STATUS REPORT - {datetime.fromtimestamp(self.timestamp)}

NODES: {len(self.nodes)} active
{self._format_nodes()}

CONNECTIONS: {len(self.connections)} links
{self._format_connections()}

PERFORMANCE:
{self._format_performance()}

SECURITY:
{self._format_security()}

RECOMMENDATIONS NEEDED FOR:
- Route optimization
- Load balancing
- Security response
- Resource allocation
"""
    
    def _format_nodes(self) -> str:
        """Format node information"""
        if not self.nodes:
            return "No nodes available"
        
        lines = []
        for node_id, node_info in list(self.nodes.items())[:5]:  # Top 5
            status = node_info.get('status', 'unknown')
            load = node_info.get('cpu_usage', 0)
            lines.append(f"  {node_id}: {status} (load: {load:.1f}%)")
        
        if len(self.nodes) > 5:
            lines.append(f"  ... and {len(self.nodes) - 5} more")
        
        return "\n".join(lines)
    
    def _format_connections(self) -> str:
        """Format connection information"""
        if not self.connections:
            return "No connections available"
        
        lines = []
        for conn_id, conn_info in list(self.connections.items())[:3]:
            latency = conn_info.get('latency', 0)
            bandwidth = conn_info.get('bandwidth', 0)
            lines.append(f"  {conn_id}: {latency:.1f}ms, {bandwidth:.1f}Mbps")
        
        return "\n".join(lines)
    
    def _format_performance(self) -> str:
        """Format performance metrics"""
        metrics = self.performance_metrics
        return f"""  Throughput: {metrics.get('throughput', 0):.1f} msg/s
  Latency P95: {metrics.get('latency_p95', 0):.1f}ms
  Error Rate: {metrics.get('error_rate', 0):.2f}%
  CPU Usage: {metrics.get('cpu_usage', 0):.1f}%
  Memory Usage: {metrics.get('memory_usage', 0):.1f}%"""
    
    def _format_security(self) -> str:
        """Format security information"""
        if not self.security_alerts:
            return "No active security alerts"
        
        return f"{len(self.security_alerts)} active alerts"

# ============================================================================
# AI MODEL IMPLEMENTATIONS
# ============================================================================

class OllamaBackend:
    """Ollama backend for running local models"""
    
    def __init__(self, config: AIModelConfig):
        self.config = config
        self.api_url = config.api_url
        self.model_name = config.model_name
        
    async def generate_decision(self, network_state: NetworkState, 
                              decision_context: str) -> NetworkDecision:
        """Generate network management decision using Ollama"""
        try:
            import aiohttp
            
            # Create prompt
            prompt = self._create_decision_prompt(network_state, decision_context)
            
            # Call Ollama API
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Low for consistent decisions
                        "top_p": 0.9,
                        "max_tokens": 256
                    }
                }
                
                async with session.post(
                    f"{self.api_url}/api/generate",
                    json=payload,
                    timeout=30
                ) as response:
                    result = await response.json()
                    decision_text = result.get("response", "")
                    
                    return self._parse_decision(decision_text)
                    
        except Exception as e:
            logger.error(f"Ollama decision generation failed: {e}")
            return self._create_fallback_decision(network_state)
    
    def _create_decision_prompt(self, network_state: NetworkState, 
                               context: str) -> str:
        """Create optimized prompt for network decisions"""
        return f"""You are an AI network controller. Analyze the network state and provide ONE specific action.

{network_state.to_prompt()}

CONTEXT: {context}

Respond ONLY in this JSON format:
{{
  "action": "route_optimization|load_balancing|security_response|resource_scaling",
  "parameters": {{"key": "value"}},
  "confidence": 0.85,
  "reasoning": "brief explanation"
}}

Decision:"""
    
    def _parse_decision(self, decision_text: str) -> NetworkDecision:
        """Parse AI model response into NetworkDecision"""
        try:
            # Extract JSON from response
            start = decision_text.find('{')
            end = decision_text.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = decision_text[start:end]
                decision_data = json.loads(json_str)
                
                return NetworkDecision(
                    decision_type=decision_data.get("action", "no_action"),
                    parameters=decision_data.get("parameters", {}),
                    confidence=decision_data.get("confidence", 0.5),
                    reasoning=decision_data.get("reasoning", "No reasoning provided")
                )
        except Exception as e:
            logger.error(f"Failed to parse decision: {e}")
        
        return NetworkDecision("no_action", {}, 0.1, "Failed to parse decision")
    
    def _create_fallback_decision(self, network_state: NetworkState) -> NetworkDecision:
        """Create conservative fallback decision"""
        return NetworkDecision(
            decision_type="monitor",
            parameters={"action": "continue_monitoring"},
            confidence=0.8,
            reasoning="Fallback decision due to AI model unavailability"
        )

class TransformersBackend:
    """Hugging Face Transformers backend for local inference"""
    
    def __init__(self, config: AIModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = "cpu"  # Use CPU for lightweight deployment
        
    async def initialize(self):
        """Initialize the model"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            model_name = self.config.model_name or "microsoft/DialoGPT-small"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map=self.device
            )
            
            # Add special tokens for network management
            special_tokens = {
                "additional_special_tokens": [
                    "[NETWORK_STATE]", "[DECISION]", "[CONFIDENCE]", "[REASONING]"
                ]
            }
            self.tokenizer.add_special_tokens(special_tokens)
            self.model.resize_token_embeddings(len(self.tokenizer))
            
            logger.info(f"Initialized Transformers model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Transformers model: {e}")
            raise
    
    async def generate_decision(self, network_state: NetworkState, 
                              decision_context: str) -> NetworkDecision:
        """Generate decision using Transformers model"""
        if not self.model:
            await self.initialize()
        
        try:
            prompt = self._create_structured_prompt(network_state, decision_context)
            
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 128,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            decision_part = response[len(prompt):].strip()
            
            return self._parse_structured_decision(decision_part)
            
        except Exception as e:
            logger.error(f"Transformers decision generation failed: {e}")
            return self._create_fallback_decision()
    
    def _create_structured_prompt(self, network_state: NetworkState, 
                                 context: str) -> str:
        """Create structured prompt for fine-tuned model"""
        return f"""[NETWORK_STATE]
Nodes: {len(network_state.nodes)}
Connections: {len(network_state.connections)}
CPU: {network_state.performance_metrics.get('cpu_usage', 0):.1f}%
Memory: {network_state.performance_metrics.get('memory_usage', 0):.1f}%
Latency: {network_state.performance_metrics.get('latency_p95', 0):.1f}ms
Context: {context}

[DECISION]"""
    
    def _parse_structured_decision(self, decision_text: str) -> NetworkDecision:
        """Parse structured decision response"""
        # Simple pattern matching for fine-tuned model responses
        lines = decision_text.split('\n')
        
        action = "monitor"
        parameters = {}
        confidence = 0.5
        reasoning = "Default monitoring action"
        
        for line in lines:
            line = line.strip()
            if line.startswith("Action:"):
                action = line.split(":", 1)[1].strip()
            elif line.startswith("Confidence:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith("Reasoning:"):
                reasoning = line.split(":", 1)[1].strip()
        
        return NetworkDecision(action, parameters, confidence, reasoning)
    
    def _create_fallback_decision(self) -> NetworkDecision:
        """Create fallback decision"""
        return NetworkDecision("monitor", {}, 0.7, "Transformers fallback")

# ============================================================================
# MAIN AI NETWORK CONTROLLER
# ============================================================================

class AINetworkController:
    """Main AI controller for autonomous network management"""
    
    def __init__(self, config: AIModelConfig, network_config: NetworkConfig):
        self.config = config
        self.network_config = network_config
        
        # Initialize backend
        if config.backend == ModelBackend.OLLAMA:
            self.backend = OllamaBackend(config)
        elif config.backend == ModelBackend.TRANSFORMERS:
            self.backend = TransformersBackend(config)
        else:
            raise ValueError(f"Unsupported backend: {config.backend}")
        
        # State management
        self.network_state = NetworkState()
        self.decision_history = []
        self.training_data = []
        
        # Decision execution
        self.decision_executors = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Monitoring
        self.performance_tracker = {}
        self.running = False
        self.decision_task = None
        
        self._setup_decision_executors()
    
    def _setup_decision_executors(self):
        """Setup decision execution handlers"""
        self.decision_executors = {
            "route_optimization": self._execute_route_optimization,
            "load_balancing": self._execute_load_balancing,
            "security_response": self._execute_security_response,
            "resource_scaling": self._execute_resource_scaling,
            "connection_management": self._execute_connection_management,
            "monitor": self._execute_monitoring
        }
    
    async def start(self):
        """Start the AI network controller"""
        logger.info("Starting AI Network Controller...")
        
        self.running = True
        
        # Initialize backend if needed
        if hasattr(self.backend, 'initialize'):
            await self.backend.initialize()
        
        # Start decision loop
        self.decision_task = asyncio.create_task(self._decision_loop())
        
        logger.info("AI Network Controller started successfully")
    
    async def stop(self):
        """Stop the AI network controller"""
        logger.info("Stopping AI Network Controller...")
        
        self.running = False
        
        if self.decision_task:
            self.decision_task.cancel()
            try:
                await self.decision_task
            except asyncio.CancelledError:
                pass
        
        self.executor.shutdown(wait=True)
        
        logger.info("AI Network Controller stopped")
    
    async def _decision_loop(self):
        """Main decision making loop"""
        while self.running:
            try:
                # Update network state
                await self._update_network_state()
                
                # Determine decision context
                context = self._analyze_context()
                
                # Generate AI decision
                decision = await self.backend.generate_decision(
                    self.network_state, context
                )
                
                # Execute decision if confidence is high enough
                if decision.confidence > 0.6:
                    await self._execute_decision(decision)
                
                # Store for training
                if self.config.collect_training_data:
                    self._collect_training_sample(decision)
                
                # Wait for next decision cycle
                await asyncio.sleep(self.config.decision_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in decision loop: {e}")
                await asyncio.sleep(5)  # Back off on error
    
    async def _update_network_state(self):
        """Update current network state from all sources"""
        # This would integrate with your existing network components
        # For now, simulate state updates
        
        self.network_state.timestamp = time.time()
        
        # Update performance metrics (integrate with your monitoring)
        self.network_state.performance_metrics = {
            'throughput': np.random.normal(1000, 100),
            'latency_p95': np.random.normal(50, 10),
            'error_rate': np.random.exponential(0.01),
            'cpu_usage': np.random.normal(60, 15),
            'memory_usage': np.random.normal(45, 10)
        }
        
        # Update node information
        # self.network_state.nodes = await self.get_node_states()
        
        # Update connection information  
        # self.network_state.connections = await self.get_connection_states()
    
    def _analyze_context(self) -> str:
        """Analyze current situation to provide context for AI decisions"""
        contexts = []
        
        # Performance context
        metrics = self.network_state.performance_metrics
        
        if metrics.get('error_rate', 0) > 0.05:
            contexts.append("High error rate detected")
        
        if metrics.get('latency_p95', 0) > 100:
            contexts.append("High latency detected")
        
        if metrics.get('cpu_usage', 0) > 80:
            contexts.append("High CPU usage")
        
        if metrics.get('throughput', 0) < 500:
            contexts.append("Low throughput")
        
        # Security context
        if self.network_state.security_alerts:
            contexts.append(f"{len(self.network_state.security_alerts)} security alerts")
        
        # Default context
        if not contexts:
            contexts.append("Normal operation")
        
        return "; ".join(contexts)
    
    async def _execute_decision(self, decision: NetworkDecision):
        """Execute a network management decision"""
        logger.info(f"Executing decision: {decision.decision_type} "
                   f"(confidence: {decision.confidence:.2f})")
        
        executor = self.decision_executors.get(decision.decision_type)
        if not executor:
            logger.warning(f"No executor for decision type: {decision.decision_type}")
            return
        
        try:
            # Execute decision
            result = await executor(decision.parameters)
            decision.executed = True
            decision.result = result
            
            # Track decision in history
            self.decision_history.append(decision)
            
            # Keep only recent decisions
            if len(self.decision_history) > 1000:
                self.decision_history = self.decision_history[-500:]
            
            logger.info(f"Decision executed successfully: {result}")
            
        except Exception as e:
            logger.error(f"Failed to execute decision {decision.decision_type}: {e}")
            decision.result = {"error": str(e)}
    
    # Decision Executors
    async def _execute_route_optimization(self, parameters: Dict) -> Dict:
        """Execute route optimization decision"""
        # Integrate with your routing system
        logger.info("Optimizing network routes based on AI recommendation")
        return {"status": "routes_optimized", "affected_routes": 5}
    
    async def _execute_load_balancing(self, parameters: Dict) -> Dict:
        """Execute load balancing decision"""
        # Integrate with your load balancing system
        logger.info("Adjusting load balancing based on AI recommendation")
        return {"status": "load_balanced", "redistributed_connections": 10}
    
    async def _execute_security_response(self, parameters: Dict) -> Dict:
        """Execute security response decision"""
        # Integrate with your security system
        logger.info("Executing security response based on AI recommendation")
        return {"status": "security_measures_applied", "blocked_ips": 2}
    
    async def _execute_resource_scaling(self, parameters: Dict) -> Dict:
        """Execute resource scaling decision"""
        # Integrate with your resource management
        logger.info("Scaling resources based on AI recommendation")
        return {"status": "resources_scaled", "new_capacity": "110%"}
    
    async def _execute_connection_management(self, parameters: Dict) -> Dict:
        """Execute connection management decision"""
        # Integrate with your connection pool
        logger.info("Managing connections based on AI recommendation")
        return {"status": "connections_managed", "optimized_connections": 15}
    
    async def _execute_monitoring(self, parameters: Dict) -> Dict:
        """Execute monitoring decision (no-op)"""
        return {"status": "monitoring", "action": "continue"}
    
    def _collect_training_sample(self, decision: NetworkDecision):
        """Collect training sample for model improvement"""
        sample = {
            "timestamp": time.time(),
            "network_state": {
                "nodes": len(self.network_state.nodes),
                "connections": len(self.network_state.connections),
                "performance": self.network_state.performance_metrics,
                "alerts": len(self.network_state.security_alerts)
            },
            "decision": {
                "type": decision.decision_type,
                "parameters": decision.parameters,
                "confidence": decision.confidence,
                "reasoning": decision.reasoning
            },
            "outcome": decision.result if decision.executed else None
        }
        
        self.training_data.append(sample)
        
        # Periodically save training data
        if len(self.training_data) % 100 == 0:
            asyncio.create_task(self._save_training_data())
    
    async def _save_training_data(self):
        """Save training data to file"""
        try:
            with open(self.config.training_data_file, 'a') as f:
                for sample in self.training_data[-100:]:  # Save last 100 samples
                    f.write(json.dumps(sample) + '\n')
            
            logger.debug(f"Saved {len(self.training_data)} training samples")
            
        except Exception as e:
            logger.error(f"Failed to save training data: {e}")
    
    async def fine_tune_model(self, training_data_file: Optional[str] = None):
        """Fine-tune the AI model with collected network data"""
        data_file = training_data_file or self.config.training_data_file
        
        if not Path(data_file).exists():
            logger.warning(f"Training data file not found: {data_file}")
            return
        
        logger.info("Starting model fine-tuning...")
        
        # Load training data
        training_samples = []
        with open(data_file, 'r') as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())
                    training_samples.append(sample)
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"Loaded {len(training_samples)} training samples")
        
        # Prepare training data for your specific model backend
        if self.config.backend == ModelBackend.TRANSFORMERS:
            await self._fine_tune_transformers_model(training_samples)
        elif self.config.backend == ModelBackend.OLLAMA:
            await self._fine_tune_ollama_model(training_samples)
        
        logger.info("Model fine-tuning completed")
    
    async def _fine_tune_transformers_model(self, training_samples: List[Dict]):
        """Fine-tune Transformers model"""
        try:
            from transformers import TrainingArguments, Trainer
            
            # Convert samples to training format
            training_texts = []
            for sample in training_samples:
                # Create training text from sample
                text = self._sample_to_training_text(sample)
                training_texts.append(text)
            
            # Tokenize
            train_encodings = self.backend.tokenizer(
                training_texts,
                truncation=True,
                padding=True,
                max_length=self.config.max_sequence_length,
                return_tensors="pt"
            )
            
            # Setup training
            training_args = TrainingArguments(
                output_dir="./fine_tuned_network_model",
                num_train_epochs=self.config.fine_tune_epochs,
                per_device_train_batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate,
                logging_steps=100,
                save_steps=500,
            )
            
            # Create custom dataset
            class NetworkDataset:
                def __init__(self, encodings):
                    self.encodings = encodings
                
                def __getitem__(self, idx):
                    return {key: val[idx] for key, val in self.encodings.items()}
                
                def __len__(self):
                    return len(self.encodings['input_ids'])
            
            train_dataset = NetworkDataset(train_encodings)
            
            # Create trainer
            trainer = Trainer(
                model=self.backend.model,
                args=training_args,
                train_dataset=train_dataset,
            )
            
            # Train
            trainer.train()
            
            # Save model
            trainer.save_model("./fine_tuned_network_model")
            
            logger.info("Transformers model fine-tuning completed")
            
        except Exception as e:
            logger.error(f"Failed to fine-tune Transformers model: {e}")
    
    async def _fine_tune_ollama_model(self, training_samples: List[Dict]):
        """Fine-tune Ollama model (create modelfile)"""
        try:
            # Create training data in Ollama format
            modelfile_content = f"""FROM {self.config.model_name}

PARAMETER temperature 0.3
PARAMETER top_p 0.9

SYSTEM You are an AI network controller. Analyze network states and provide specific management decisions in JSON format.

"""
            
            # Add training examples
            for sample in training_samples[:100]:  # Limit for Ollama
                prompt = self._sample_to_training_text(sample)
                modelfile_content += f'MESSAGE user "{prompt}"\n'
                modelfile_content += f'MESSAGE assistant "{json.dumps(sample["decision"])}"\n\n'
            
            # Save modelfile
            with open("NetworkAI.modelfile", "w") as f:
                f.write(modelfile_content)
            
            logger.info("Created Ollama modelfile for fine-tuning")
            logger.info("Run: ollama create NetworkAI -f NetworkAI.modelfile")
            
        except Exception as e:
            logger.error(f"Failed to create Ollama modelfile: {e}")
    
    def _sample_to_training_text(self, sample: Dict) -> str:
        """Convert training sample to text format"""
        network_state = sample["network_state"]
        return f"""Network State:
Nodes: {network_state.get("nodes", 0)}
Connections: {network_state.get("connections", 0)}
CPU: {network_state.get("performance", {}).get("cpu_usage", 0):.1f}%
Memory: {network_state.get("performance", {}).get("memory_usage", 0):.1f}%
Latency: {network_state.get("performance", {}).get("latency_p95", 0):.1f}ms
Alerts: {network_state.get("alerts", 0)}

Required Decision:"""

# ============================================================================
# INTEGRATION HELPER
# ============================================================================

class NetworkAIManager:
    """High-level manager for AI-driven network control"""
    
    def __init__(self, network_config: NetworkConfig):
        self.network_config = network_config
        
        # Default AI configuration for small, efficient model
        self.ai_config = AIModelConfig(
            backend=ModelBackend.OLLAMA,
            model_name="phi3:mini",  # 3.8B parameters, good for network tasks
            decision_interval=10.0,  # Make decisions every 10 seconds
            collect_training_data=True
        )
        
        self.controller = AINetworkController(self.ai_config, network_config)
        self.running = False
    
    async def start_ai_control(self):
        """Start AI-driven network control"""
        logger.info("Starting AI Network Management...")
        
        self.running = True
        await self.controller.start()
        
        logger.info("AI Network Management is now active")
    
    async def stop_ai_control(self):
        """Stop AI-driven network control"""
        logger.info("Stopping AI Network Management...")
        
        self.running = False
        await self.controller.stop()
        
        logger.info("AI Network Management stopped")
    
    async def fine_tune_network_ai(self):
        """Fine-tune the AI model with network data"""
        await self.controller.fine_tune_model()
    
    def get_ai_status(self) -> Dict[str, Any]:
        """Get current AI controller status"""
        return {
            "running": self.running,
            "backend": self.ai_config.backend.value,
            "model": self.ai_config.model_name,
            "decisions_made": len(self.controller.decision_history),
            "training_samples": len(self.controller.training_data),
            "last_decision": self.controller.decision_history[-1].__dict__ if self.controller.decision_history else None
        }

# Example usage function
async def create_ai_managed_network(network_config: NetworkConfig) -> NetworkAIManager:
    """Create and start an AI-managed network"""
    ai_manager = NetworkAIManager(network_config)
    await ai_manager.start_ai_control()
    return ai_manager
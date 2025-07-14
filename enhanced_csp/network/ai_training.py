# enhanced_csp/network/ai_training.py
"""
AI Model Fine-Tuning System for Network Management
=================================================
Complete system for collecting training data and fine-tuning small AI models
"""

import asyncio
import json
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timedelta
import random
import pickle

logger = logging.getLogger(__name__)

# ============================================================================
# TRAINING DATA GENERATION
# ============================================================================

@dataclass
class NetworkScenario:
    """Network scenario for training data generation"""
    name: str
    description: str
    node_count: int
    connection_density: float  # 0.0 to 1.0
    load_pattern: str  # "low", "medium", "high", "spike", "oscillating"
    failure_rate: float  # 0.0 to 1.0
    security_threats: List[str]
    optimal_actions: List[str]

class TrainingDataGenerator:
    """Generate synthetic and real training data for network AI"""
    
    def __init__(self):
        self.scenarios = self._create_base_scenarios()
        self.decision_templates = self._create_decision_templates()
        
    def _create_base_scenarios(self) -> List[NetworkScenario]:
        """Create base network scenarios for training"""
        return [
            NetworkScenario(
                name="normal_operation",
                description="Normal network operation with standard load",
                node_count=50,
                connection_density=0.3,
                load_pattern="medium",
                failure_rate=0.01,
                security_threats=[],
                optimal_actions=["monitor", "maintain_routes"]
            ),
            NetworkScenario(
                name="high_load",
                description="High traffic load requiring optimization",
                node_count=50,
                connection_density=0.3,
                load_pattern="high",
                failure_rate=0.02,
                security_threats=[],
                optimal_actions=["load_balancing", "route_optimization"]
            ),
            NetworkScenario(
                name="node_failures",
                description="Multiple node failures requiring rerouting",
                node_count=45,
                connection_density=0.25,
                load_pattern="medium",
                failure_rate=0.15,
                security_threats=[],
                optimal_actions=["route_optimization", "failover_activation"]
            ),
            NetworkScenario(
                name="ddos_attack",
                description="DDoS attack requiring security response",
                node_count=50,
                connection_density=0.3,
                load_pattern="spike",
                failure_rate=0.05,
                security_threats=["ddos", "bandwidth_exhaustion"],
                optimal_actions=["security_response", "traffic_filtering", "rate_limiting"]
            ),
            NetworkScenario(
                name="congestion",
                description="Network congestion requiring load redistribution",
                node_count=50,
                connection_density=0.4,
                load_pattern="high",
                failure_rate=0.03,
                security_threats=[],
                optimal_actions=["load_balancing", "connection_management", "qos_adjustment"]
            ),
            NetworkScenario(
                name="rapid_scaling",
                description="Rapid network growth requiring resource scaling",
                node_count=75,
                connection_density=0.2,
                load_pattern="oscillating",
                failure_rate=0.02,
                security_threats=[],
                optimal_actions=["resource_scaling", "topology_optimization"]
            )
        ]
    
    def _create_decision_templates(self) -> Dict[str, Dict]:
        """Create decision templates for different scenarios"""
        return {
            "route_optimization": {
                "action": "route_optimization",
                "parameters": {
                    "algorithm": "batman_enhanced",
                    "recalculate_interval": 30,
                    "consider_latency": True,
                    "consider_bandwidth": True
                },
                "confidence": 0.85,
                "reasoning": "Network topology or performance metrics indicate suboptimal routing paths"
            },
            "load_balancing": {
                "action": "load_balancing",
                "parameters": {
                    "strategy": "adaptive_weighted",
                    "rebalance_threshold": 0.7,
                    "consider_node_capacity": True
                },
                "confidence": 0.90,
                "reasoning": "Uneven load distribution detected across network nodes"
            },
            "security_response": {
                "action": "security_response",
                "parameters": {
                    "response_level": "medium",
                    "enable_filtering": True,
                    "quarantine_suspicious": True,
                    "alert_administrators": True
                },
                "confidence": 0.95,
                "reasoning": "Security threat detected requiring immediate response"
            },
            "resource_scaling": {
                "action": "resource_scaling",
                "parameters": {
                    "scale_direction": "up",
                    "scale_factor": 1.2,
                    "components": ["connections", "bandwidth"]
                },
                "confidence": 0.80,
                "reasoning": "Resource utilization approaching capacity limits"
            },
            "connection_management": {
                "action": "connection_management",
                "parameters": {
                    "optimize_pools": True,
                    "close_idle": True,
                    "rebalance_connections": True
                },
                "confidence": 0.75,
                "reasoning": "Connection pool optimization needed for better resource utilization"
            },
            "monitor": {
                "action": "monitor",
                "parameters": {
                    "continue_monitoring": True
                },
                "confidence": 0.70,
                "reasoning": "Network state is stable, continue monitoring"
            }
        }
    
    def generate_training_sample(self, scenario: NetworkScenario) -> Dict[str, Any]:
        """Generate a single training sample from a scenario"""
        # Generate realistic network state
        network_state = self._generate_network_state(scenario)
        
        # Determine optimal decision based on scenario
        optimal_decision = self._determine_optimal_decision(scenario, network_state)
        
        # Create training sample
        sample = {
            "timestamp": time.time(),
            "scenario": scenario.name,
            "network_state": network_state,
            "decision": optimal_decision,
            "outcome": self._simulate_outcome(optimal_decision, scenario)
        }
        
        return sample
    
    def _generate_network_state(self, scenario: NetworkScenario) -> Dict[str, Any]:
        """Generate realistic network state for scenario"""
        # Base metrics influenced by scenario
        load_multiplier = {
            "low": 0.3,
            "medium": 0.6,
            "high": 0.85,
            "spike": 0.95,
            "oscillating": random.uniform(0.3, 0.9)
        }.get(scenario.load_pattern, 0.6)
        
        # Generate performance metrics
        base_cpu = 40 + (load_multiplier * 40)
        base_memory = 30 + (load_multiplier * 50)
        base_latency = 20 + (load_multiplier * 80)
        base_throughput = 1000 * (1.2 - load_multiplier)
        
        # Add realistic noise
        cpu_usage = max(0, min(100, np.random.normal(base_cpu, 10)))
        memory_usage = max(0, min(100, np.random.normal(base_memory, 8)))
        latency_p95 = max(5, np.random.normal(base_latency, 15))
        throughput = max(100, np.random.normal(base_throughput, 200))
        error_rate = np.random.exponential(scenario.failure_rate)
        
        # Security alerts based on threats
        security_alerts = len(scenario.security_threats) + np.random.poisson(0.1)
        
        return {
            "nodes": scenario.node_count + np.random.randint(-5, 6),
            "connections": int(scenario.node_count * scenario.connection_density),
            "performance": {
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "latency_p95": latency_p95,
                "throughput": throughput,
                "error_rate": min(1.0, error_rate),
                "bandwidth_utilization": load_multiplier + np.random.normal(0, 0.1)
            },
            "security": {
                "alerts": int(security_alerts),
                "threats": scenario.security_threats,
                "threat_level": "high" if scenario.security_threats else "low"
            },
            "topology": {
                "density": scenario.connection_density,
                "avg_path_length": 2 + np.random.uniform(0, 2),
                "clustering_coefficient": scenario.connection_density * 0.7
            }
        }
    
    def _determine_optimal_decision(self, scenario: NetworkScenario, 
                                  network_state: Dict) -> Dict[str, Any]:
        """Determine optimal decision for given scenario and state"""
        # Use scenario's optimal actions as guidance
        if scenario.optimal_actions:
            primary_action = random.choice(scenario.optimal_actions)
        else:
            primary_action = "monitor"
        
        # Get decision template
        decision_template = self.decision_templates.get(primary_action, 
                                                       self.decision_templates["monitor"])
        
        # Adjust decision based on current state
        decision = decision_template.copy()
        
        # Adjust confidence based on state clarity
        performance = network_state["performance"]
        
        if performance["cpu_usage"] > 90 or performance["error_rate"] > 0.1:
            decision["confidence"] = min(0.95, decision["confidence"] + 0.1)
        elif performance["cpu_usage"] < 30 and performance["error_rate"] < 0.01:
            decision["confidence"] = max(0.6, decision["confidence"] - 0.1)
        
        # Adjust parameters based on severity
        if primary_action == "route_optimization" and performance["latency_p95"] > 100:
            decision["parameters"]["priority"] = "high"
            decision["parameters"]["recalculate_interval"] = 15
        
        return decision
    
    def _simulate_outcome(self, decision: Dict, scenario: NetworkScenario) -> Dict[str, Any]:
        """Simulate the outcome of a decision"""
        action = decision.get("action", "monitor")
        
        # Simulate realistic outcomes
        if action == "route_optimization":
            return {
                "latency_improvement": np.random.uniform(10, 30),
                "throughput_improvement": np.random.uniform(5, 20),
                "routes_updated": np.random.randint(3, 15),
                "success": True
            }
        elif action == "load_balancing":
            return {
                "load_variance_reduction": np.random.uniform(15, 40),
                "connections_rebalanced": np.random.randint(10, 50),
                "cpu_usage_reduction": np.random.uniform(5, 20),
                "success": True
            }
        elif action == "security_response":
            return {
                "threats_mitigated": len(scenario.security_threats),
                "blocked_connections": np.random.randint(5, 50),
                "security_level_improvement": "high" if scenario.security_threats else "medium",
                "success": True
            }
        else:
            return {
                "action_taken": action,
                "success": True
            }
    
    def generate_training_dataset(self, num_samples: int = 10000) -> List[Dict[str, Any]]:
        """Generate complete training dataset"""
        logger.info(f"Generating {num_samples} training samples...")
        
        dataset = []
        
        for i in range(num_samples):
            # Select random scenario (weighted by importance)
            scenario_weights = [3, 2, 2, 1, 2, 1]  # More normal operation examples
            scenario = random.choices(self.scenarios, weights=scenario_weights)[0]
            
            # Generate sample
            sample = self.generate_training_sample(scenario)
            dataset.append(sample)
            
            if (i + 1) % 1000 == 0:
                logger.info(f"Generated {i + 1}/{num_samples} samples")
        
        logger.info("Training dataset generation complete")
        return dataset

# ============================================================================
# MODEL FINE-TUNING SYSTEM
# ============================================================================

class NetworkAITrainer:
    """Complete system for training network AI models"""
    
    def __init__(self, output_dir: str = "./trained_models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.data_generator = TrainingDataGenerator()
        self.training_data = []
        
    def prepare_training_data(self, num_samples: int = 10000, 
                            include_real_data: bool = True) -> str:
        """Prepare comprehensive training data"""
        logger.info("Preparing training data...")
        
        # Generate synthetic data
        synthetic_data = self.data_generator.generate_training_dataset(num_samples)
        self.training_data.extend(synthetic_data)
        
        # Load real data if available
        if include_real_data:
            real_data = self._load_real_network_data()
            if real_data:
                self.training_data.extend(real_data)
                logger.info(f"Added {len(real_data)} real network samples")
        
        # Save training data
        data_file = self.output_dir / "network_training_data.jsonl"
        with open(data_file, 'w') as f:
            for sample in self.training_data:
                f.write(json.dumps(sample, default=str) + '\n')
        
        logger.info(f"Saved {len(self.training_data)} training samples to {data_file}")
        return str(data_file)
    
    def _load_real_network_data(self) -> List[Dict]:
        """Load real network data if available"""
        real_data_file = Path("real_network_data.jsonl")
        if not real_data_file.exists():
            return []
        
        try:
            real_data = []
            with open(real_data_file, 'r') as f:
                for line in f:
                    try:
                        sample = json.loads(line.strip())
                        real_data.append(sample)
                    except json.JSONDecodeError:
                        continue
            
            logger.info(f"Loaded {len(real_data)} real network samples")
            return real_data
            
        except Exception as e:
            logger.error(f"Failed to load real network data: {e}")
            return []
    
    def create_ollama_modelfile(self, base_model: str = "phi3:mini") -> str:
        """Create Ollama modelfile for fine-tuning"""
        logger.info("Creating Ollama modelfile...")
        
        modelfile_content = f"""FROM {base_model}

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER stop "\\n\\n"

SYSTEM \"\"\"You are NetworkAI, an expert AI system for managing computer networks. You analyze network states and provide specific management decisions.

Your responses must be in this exact JSON format:
{{
  "action": "route_optimization|load_balancing|security_response|resource_scaling|connection_management|monitor",
  "parameters": {{"key": "value"}},
  "confidence": 0.85,
  "reasoning": "brief explanation"
}}

Always provide actionable decisions based on network metrics like CPU usage, memory usage, latency, throughput, error rates, and security alerts.\"\"\"

"""
        
        # Add training examples (limit for Ollama)
        training_examples = self.training_data[:200]  # Ollama limitation
        
        for sample in training_examples:
            network_state = sample["network_state"]
            decision = sample["decision"]
            
            # Create prompt
            prompt = self._create_training_prompt(network_state)
            response = json.dumps(decision, indent=2)
            
            modelfile_content += f'MESSAGE user """{prompt}"""\n'
            modelfile_content += f'MESSAGE assistant """{response}"""\n\n'
        
        # Save modelfile
        modelfile_path = self.output_dir / "NetworkAI.modelfile"
        with open(modelfile_path, 'w') as f:
            f.write(modelfile_content)
        
        logger.info(f"Created Ollama modelfile: {modelfile_path}")
        logger.info(f"To use: ollama create NetworkAI -f {modelfile_path}")
        
        return str(modelfile_path)
    
    def _create_training_prompt(self, network_state: Dict) -> str:
        """Create training prompt from network state"""
        perf = network_state.get("performance", {})
        security = network_state.get("security", {})
        
        return f"""NETWORK STATUS:
Nodes: {network_state.get("nodes", 0)}
Connections: {network_state.get("connections", 0)}
CPU Usage: {perf.get("cpu_usage", 0):.1f}%
Memory Usage: {perf.get("memory_usage", 0):.1f}%
Latency P95: {perf.get("latency_p95", 0):.1f}ms
Throughput: {perf.get("throughput", 0):.1f} msg/s
Error Rate: {perf.get("error_rate", 0):.3f}
Security Alerts: {security.get("alerts", 0)}
Threat Level: {security.get("threat_level", "low")}

What action should be taken?"""
    
    def create_huggingface_dataset(self) -> str:
        """Create dataset for Hugging Face fine-tuning"""
        logger.info("Creating Hugging Face dataset...")
        
        # Prepare data in text format
        training_texts = []
        for sample in self.training_data:
            network_state = sample["network_state"]
            decision = sample["decision"]
            
            prompt = self._create_training_prompt(network_state)
            response = json.dumps(decision)
            
            # Create training text with special tokens
            training_text = f"<|startoftext|>{prompt}<|sep|>{response}<|endoftext|>"
            training_texts.append(training_text)
        
        # Save as text file
        dataset_path = self.output_dir / "hf_training_data.txt"
        with open(dataset_path, 'w') as f:
            for text in training_texts:
                f.write(text + '\n')
        
        logger.info(f"Created Hugging Face dataset: {dataset_path}")
        return str(dataset_path)
    
    def create_openai_fine_tuning_data(self) -> str:
        """Create data for OpenAI fine-tuning format"""
        logger.info("Creating OpenAI fine-tuning dataset...")
        
        openai_data = []
        for sample in self.training_data:
            network_state = sample["network_state"]
            decision = sample["decision"]
            
            prompt = self._create_training_prompt(network_state)
            response = json.dumps(decision)
            
            openai_sample = {
                "messages": [
                    {"role": "system", "content": "You are NetworkAI, an expert network management system."},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ]
            }
            openai_data.append(openai_sample)
        
        # Save as JSONL
        dataset_path = self.output_dir / "openai_fine_tuning_data.jsonl"
        with open(dataset_path, 'w') as f:
            for sample in openai_data:
                f.write(json.dumps(sample) + '\n')
        
        logger.info(f"Created OpenAI fine-tuning dataset: {dataset_path}")
        return str(dataset_path)
    
    async def run_complete_training_pipeline(self, num_samples: int = 10000):
        """Run complete training pipeline"""
        logger.info("Starting complete AI training pipeline...")
        
        # Step 1: Prepare training data
        data_file = self.prepare_training_data(num_samples)
        
        # Step 2: Create model files for different platforms
        ollama_modelfile = self.create_ollama_modelfile()
        hf_dataset = self.create_huggingface_dataset()
        openai_dataset = self.create_openai_fine_tuning_data()
        
        # Step 3: Create training scripts
        self._create_training_scripts()
        
        # Step 4: Generate evaluation metrics
        self._create_evaluation_script()
        
        logger.info("Training pipeline complete!")
        logger.info(f"Files created in: {self.output_dir}")
        logger.info("Next steps:")
        logger.info("1. For Ollama: ollama create NetworkAI -f NetworkAI.modelfile")
        logger.info("2. For Hugging Face: Use the training script")
        logger.info("3. For OpenAI: Upload the JSONL file")
    
    def _create_training_scripts(self):
        """Create training scripts for different platforms"""
        
        # Hugging Face training script
        hf_script = """#!/usr/bin/env python3
# Hugging Face fine-tuning script
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from datasets import Dataset
import torch

def train_network_ai():
    # Load model and tokenizer
    model_name = "microsoft/DialoGPT-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Add special tokens
    special_tokens = {"pad_token": "<|pad|>", "sep_token": "<|sep|>"}
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    # Load and tokenize data
    with open("hf_training_data.txt", "r") as f:
        texts = [line.strip() for line in f]
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)
    
    dataset = Dataset.from_dict({"text": texts})
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./network_ai_model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        warmup_steps=100,
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
        save_total_limit=2,
        prediction_loss_only=True,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )
    
    # Train
    trainer.train()
    trainer.save_model()
    
if __name__ == "__main__":
    train_network_ai()
"""
        
        with open(self.output_dir / "train_huggingface.py", 'w') as f:
            f.write(hf_script)
        
        # Make executable
        (self.output_dir / "train_huggingface.py").chmod(0o755)
    
    def _create_evaluation_script(self):
        """Create evaluation script for trained models"""
        
        eval_script = """#!/usr/bin/env python3
# Network AI model evaluation script
import json
import time
import requests
from typing import Dict, Any

def evaluate_ollama_model(model_name: str = "NetworkAI"):
    '''Evaluate Ollama model performance'''
    
    test_cases = [
        {
            "scenario": "high_cpu",
            "prompt": '''NETWORK STATUS:
Nodes: 50
Connections: 150
CPU Usage: 85.0%
Memory Usage: 60.0%
Latency P95: 120.0ms
Throughput: 800.0 msg/s
Error Rate: 0.020
Security Alerts: 0
Threat Level: low

What action should be taken?''',
            "expected_action": "load_balancing"
        },
        {
            "scenario": "security_threat",
            "prompt": '''NETWORK STATUS:
Nodes: 50
Connections: 150
CPU Usage: 45.0%
Memory Usage: 40.0%
Latency P95: 80.0ms
Throughput: 1200.0 msg/s
Error Rate: 0.050
Security Alerts: 5
Threat Level: high

What action should be taken?''',
            "expected_action": "security_response"
        }
    ]
    
    correct_predictions = 0
    
    for test_case in test_cases:
        try:
            # Call Ollama API
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model_name,
                    "prompt": test_case["prompt"],
                    "stream": False
                }
            )
            
            result = response.json()
            decision_text = result.get("response", "")
            
            # Parse decision
            try:
                decision_start = decision_text.find('{')
                decision_end = decision_text.rfind('}') + 1
                if decision_start >= 0 and decision_end > decision_start:
                    decision_json = json.loads(decision_text[decision_start:decision_end])
                    predicted_action = decision_json.get("action", "unknown")
                    
                    print(f"Scenario: {test_case['scenario']}")
                    print(f"Expected: {test_case['expected_action']}")
                    print(f"Predicted: {predicted_action}")
                    print(f"Confidence: {decision_json.get('confidence', 'N/A')}")
                    print()
                    
                    if predicted_action == test_case["expected_action"]:
                        correct_predictions += 1
                        
            except json.JSONDecodeError:
                print(f"Failed to parse decision for {test_case['scenario']}")
                
        except Exception as e:
            print(f"Error evaluating {test_case['scenario']}: {e}")
    
    accuracy = correct_predictions / len(test_cases)
    print(f"Overall Accuracy: {accuracy:.2%} ({correct_predictions}/{len(test_cases)})")

if __name__ == "__main__":
    evaluate_ollama_model()
"""
        
        with open(self.output_dir / "evaluate_model.py", 'w') as f:
            f.write(eval_script)
        
        (self.output_dir / "evaluate_model.py").chmod(0o755)

# ============================================================================
# EASY SETUP FUNCTIONS
# ============================================================================

async def setup_network_ai(network_config, model_type: str = "ollama"):
    """Easy setup function for network AI"""
    logger.info("Setting up Network AI system...")
    
    # Create trainer
    trainer = NetworkAITrainer()
    
    # Generate training data and model files
    await trainer.run_complete_training_pipeline(num_samples=5000)
    
    # Create AI manager
    from .ai_controller import NetworkAIManager, AIModelConfig, ModelBackend
    
    ai_config = AIModelConfig(
        backend=ModelBackend.OLLAMA if model_type == "ollama" else ModelBackend.TRANSFORMERS,
        model_name="NetworkAI" if model_type == "ollama" else "microsoft/DialoGPT-small"
    )
    
    ai_manager = NetworkAIManager(network_config)
    ai_manager.ai_config = ai_config
    
    logger.info("Network AI setup complete!")
    logger.info("Run the training commands shown above to complete the setup")
    
    return ai_manager

def quick_start_network_ai():
    """Quick start function with all steps"""
    print("""
🤖 Network AI Quick Start Guide
===============================

1. Generate Training Data:
   python -c "
   import asyncio
   from enhanced_csp.network.ai_training import NetworkAITrainer
   trainer = NetworkAITrainer()
   asyncio.run(trainer.run_complete_training_pipeline(5000))
   "

2. Install Ollama (recommended for local deployment):
   curl -fsSL https://ollama.ai/install.sh | sh
   
3. Create the fine-tuned model:
   ollama create NetworkAI -f ./trained_models/NetworkAI.modelfile
   
4. Start your network with AI:
   python -c "
   import asyncio
   from enhanced_csp.network.ai_training import setup_network_ai
   from enhanced_csp.network.core.config import NetworkConfig
   
   config = NetworkConfig.development()
   ai_manager = asyncio.run(setup_network_ai(config))
   asyncio.run(ai_manager.start_ai_control())
   "

5. Monitor AI decisions:
   python ./trained_models/evaluate_model.py

Your network will now be autonomously managed by AI! 🚀
""")

if __name__ == "__main__":
    quick_start_network_ai()
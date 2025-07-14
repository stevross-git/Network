#!/usr/bin/env python3
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

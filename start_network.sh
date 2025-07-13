#!/bin/bash
# Enhanced CSP Network Startup Script
# Generated automatically by NetworkConfigHelper

echo "🚀 Starting Enhanced CSP Network..."
echo "📋 Using config: config/network_config.json"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not found"
    exit 1
fi

# Check if required files exist
if [ ! -f "config/network_config.json" ]; then
    echo "❌ Configuration file not found: config/network_config.json"
    exit 1
fi

# Set environment variables
export PYTHONPATH="$PWD:$PYTHONPATH"
export CSP_NETWORK_CONFIG="config/network_config.json"

# Start the network
echo "🔧 Initializing network with configuration..."
python3 enhanced_csp/run_network.py \
    --genesis-host genesis.peoplesainetwork.com \
    --genesis-port 30300 \
    --local-port 30301 \
    --node-name csp-node \
    --log-level INFO \
    --max-connection-attempts 50

echo ""
echo "👋 Enhanced CSP Network shutdown complete"

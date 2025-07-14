# Enhanced CSP Network Overview

Created: 2025-07-13 22:39:33
Genesis Server: genesis.peoplesainetwork.com:30300
Total Nodes: 4

## Node Configuration

| Node Name | Type | Port | Capabilities | Script |
|-----------|------|------|--------------|--------|
| relay_node_1 | relay | 30400 | relay | `node_scripts/start_relay_node_1.py` |
| relay_node_2 | relay | 30401 | relay | `node_scripts/start_relay_node_2.py` |
| storage_node_1 | storage | 30402 | relay, storage | `node_scripts/start_storage_node_1.py` |
| compute_node_1 | compute | 30403 | relay, compute | `node_scripts/start_compute_node_1.py` |

## Starting the Network

### Start All Nodes
```bash
# Start each node in a separate terminal
python3 node_scripts/start_relay_node_1.py  # relay_node_1
python3 node_scripts/start_relay_node_2.py  # relay_node_2
python3 node_scripts/start_storage_node_1.py  # storage_node_1
python3 node_scripts/start_compute_node_1.py  # compute_node_1
```

### Monitor Network
```bash
# Check network status
python3 enhanced_csp/network/monitoring.py

# View network dashboard
python3 dashboard_server.py
```

## Node Types

- **Relay Nodes**: Forward messages, basic networking
- **Storage Nodes**: Provide distributed storage
- **Compute Nodes**: Execute distributed computations  
- **Super Peers**: High-capacity nodes for network stability
- **Edge Nodes**: Lightweight nodes for IoT/mobile devices

## Network Topology

The network uses adaptive mesh topology with:
- Bootstrap connections to genesis server
- Peer discovery via mDNS and DHT
- Automatic topology optimization
- BATMAN routing protocol for efficient message delivery

## Configuration Files

Node configurations are stored in `./node_configs/`
Startup scripts are stored in `./node_scripts/`

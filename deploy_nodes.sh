#!/bin/bash
# Enhanced CSP Network - Quick Node Deployment
# Create and start multiple nodes quickly

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Configuration
GENESIS_HOST="${GENESIS_HOST:-genesis.peoplesainetwork.com}"
GENESIS_PORT="${GENESIS_PORT:-30300}"
NODE_PRESET="${NODE_PRESET:-development}"

# Function to show usage
show_usage() {
    cat << EOF
Enhanced CSP Network - Quick Node Deployment

USAGE:
    $0 [OPTIONS] COMMAND

COMMANDS:
    create          Create node configurations and scripts
    start           Start all nodes
    stop            Stop all nodes
    status          Show node status
    clean           Clean up node data and configs

OPTIONS:
    --preset PRESET     Node configuration preset (development|production|testing)
    --genesis-host HOST Genesis server hostname (default: genesis.peoplesainetwork.com)
    --genesis-port PORT Genesis server port (default: 30300)
    --help              Show this help message

PRESETS:
    development     4 nodes: 2 relay, 1 storage, 1 compute
    production      9 nodes: 2 super_peer, 3 relay, 2 storage, 2 compute
    testing         2 nodes: 1 relay, 1 edge

EXAMPLES:
    $0 create                           # Create development preset
    $0 --preset production create       # Create production preset
    $0 start                           # Start all created nodes
    $0 status                          # Check node status
    $0 clean                           # Clean up everything

ENVIRONMENT VARIABLES:
    GENESIS_HOST    Genesis server hostname
    GENESIS_PORT    Genesis server port
    NODE_PRESET     Default node preset
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --preset)
            NODE_PRESET="$2"
            shift 2
            ;;
        --genesis-host)
            GENESIS_HOST="$2"
            shift 2
            ;;
        --genesis-port)
            GENESIS_PORT="$2"
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        create|start|stop|status|clean)
            COMMAND="$1"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Check if command was provided
if [[ -z "${COMMAND:-}" ]]; then
    log_error "No command specified"
    show_usage
    exit 1
fi

# Function to check dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check if we're in the right directory
    if [[ ! -d "enhanced_csp" ]]; then
        log_error "Please run this script from the root directory containing enhanced_csp/"
        exit 1
    fi
    
    log_success "Dependencies checked"
}

# Function to create nodes
create_nodes() {
    log_info "Creating nodes with preset: $NODE_PRESET"
    log_info "Genesis server: $GENESIS_HOST:$GENESIS_PORT"
    
    # Check for network_startup.py first
    if [[ ! -f "network_startup.py" ]]; then
        log_error "network_startup.py not found in current directory"
        log_info "Please ensure network_startup.py is in the same directory as this script"
        exit 1
    fi
    
    # Try the simple node creator first (more reliable)
    if [[ -f "simple_node_creator.py" ]]; then
        log_info "Using simple node creator..."
        python3 simple_node_creator.py
    elif [[ -f "node_creator.py" ]]; then
        log_info "Using advanced node creator..."
        python3 node_creator.py \
            --preset "$NODE_PRESET" \
            --genesis-host "$GENESIS_HOST" \
            --genesis-port "$GENESIS_PORT"
    else
        log_error "No node creator script found"
        log_info "Please ensure either node_creator.py or simple_node_creator.py is available"
        exit 1
    fi
    
    if [[ $? -eq 0 ]]; then
        log_success "Nodes created successfully"
        log_info "Check for generated startup scripts"
    else
        log_error "Failed to create nodes"
        exit 1
    fi
}

# Function to start all nodes
start_nodes() {
    log_info "Starting all nodes..."
    
    # Check for different types of node scripts
    if [[ -d "simple_node_scripts" ]]; then
        script_dir="simple_node_scripts"
        script_pattern="start_*.py"
    elif [[ -d "node_scripts" ]]; then
        script_dir="node_scripts" 
        script_pattern="start_*.py"
    elif [[ -f "start_all_nodes.py" ]]; then
        log_info "Using batch starter script..."
        python3 start_all_nodes.py &
        log_success "Batch starter launched"
        return
    else
        log_error "No node scripts found. Run 'create' command first."
        exit 1
    fi
    
    # Create logs directory
    mkdir -p logs
    
    # Start each node in background
    for script in "$script_dir"/$script_pattern; do
        if [[ -f "$script" ]]; then
            node_name=$(basename "$script" .py | sed 's/start_//')
            log_info "Starting node: $node_name"
            
            # Start node in background with logging
            nohup python3 "$script" > "logs/${node_name}.log" 2>&1 &
            echo $! > "logs/${node_name}.pid"
            
            log_success "Started $node_name (PID: $(cat "logs/${node_name}.pid"))"
            sleep 2  # Brief delay between starts
        fi
    done
    
    log_success "All nodes started"
    log_info "Check logs/ directory for node output"
}

# Function to stop all nodes
stop_nodes() {
    log_info "Stopping all nodes..."
    
    if [[ ! -d "logs" ]]; then
        log_warning "No running nodes found"
        return
    fi
    
    # Stop each node by PID
    for pid_file in logs/*.pid; do
        if [[ -f "$pid_file" ]]; then
            node_name=$(basename "$pid_file" .pid)
            pid=$(cat "$pid_file")
            
            if kill -0 "$pid" 2>/dev/null; then
                log_info "Stopping node: $node_name (PID: $pid)"
                kill "$pid"
                rm -f "$pid_file"
                log_success "Stopped $node_name"
            else
                log_warning "Node $node_name was not running"
                rm -f "$pid_file"
            fi
        fi
    done
    
    log_success "All nodes stopped"
}

# Function to show node status
show_status() {
    log_info "Node Status:"
    echo
    
    if [[ ! -d "logs" ]]; then
        log_warning "No node logs found"
        return
    fi
    
    printf "%-20s %-10s %-15s %-10s\n" "NODE NAME" "STATUS" "PID" "LOG SIZE"
    printf "%-20s %-10s %-15s %-10s\n" "----------" "------" "---" "--------"
    
    for pid_file in logs/*.pid; do
        if [[ -f "$pid_file" ]]; then
            node_name=$(basename "$pid_file" .pid)
            pid=$(cat "$pid_file")
            log_file="logs/${node_name}.log"
            
            if kill -0 "$pid" 2>/dev/null; then
                status="RUNNING"
                log_size=$(du -h "$log_file" 2>/dev/null | cut -f1 || echo "N/A")
            else
                status="STOPPED"
                pid="N/A"
                log_size="N/A"
                rm -f "$pid_file"  # Clean up stale PID file
            fi
            
            printf "%-20s %-10s %-15s %-10s\n" "$node_name" "$status" "$pid" "$log_size"
        fi
    done
    
    echo
    
    # Show network statistics if available
    if [[ -f "NETWORK_OVERVIEW.md" ]]; then
        total_nodes=$(grep "Total Nodes:" NETWORK_OVERVIEW.md | awk '{print $3}')
        log_info "Total configured nodes: $total_nodes"
    fi
    
    # Show recent log entries
    log_info "Recent activity (last 5 lines from each log):"
    for log_file in logs/*.log; do
        if [[ -f "$log_file" ]] && [[ -s "$log_file" ]]; then
            node_name=$(basename "$log_file" .log)
            echo
            echo "--- $node_name ---"
            tail -n 5 "$log_file" | sed 's/^/  /'
        fi
    done
}

# Function to clean up
clean_up() {
    log_warning "This will remove all node configurations, scripts, and data!"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Cleanup cancelled"
        return
    fi
    
    log_info "Cleaning up..."
    
    # Stop nodes first
    stop_nodes
    
    # Remove directories and files
    rm -rf node_configs/
    rm -rf node_scripts/
    rm -rf network_data/
    rm -rf logs/
    rm -f NETWORK_OVERVIEW.md
    
    log_success "Cleanup complete"
}

# Function to show quick help
show_quick_help() {
    cat << EOF
Quick Commands:
  $0 create           # Create nodes
  $0 start            # Start all nodes  
  $0 stop             # Stop all nodes
  $0 status           # Show status
  $0 clean            # Clean up
  $0 --help           # Full help
EOF
}

# Main execution
main() {
    echo "======================================"
    echo "Enhanced CSP Network - Node Deployment"
    echo "======================================"
    echo
    
    case $COMMAND in
        create)
            check_dependencies
            create_nodes
            ;;
        start)
            start_nodes
            ;;
        stop)
            stop_nodes
            ;;
        status)
            show_status
            ;;
        clean)
            clean_up
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            show_quick_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
# enhanced_csp/network/core/network.py
"""
EnhancedCSPNetwork - Wrapper around NetworkNode for compatibility
This provides the interface that genesis_connector expects while using the working NetworkNode.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

from .node import NetworkNode
from .config import NetworkConfig
from .types import NodeID, PeerInfo

logger = logging.getLogger(__name__)


class EnhancedCSPNetwork:
    """
    Enhanced CSP Network wrapper around NetworkNode.
    
    This class provides a high-level interface for network operations
    while delegating the actual work to NetworkNode.
    """
    
    def __init__(self, config: Optional[NetworkConfig] = None):
        self.config = config or NetworkConfig()
        self.node = NetworkNode(self.config)
        self._is_running = False
        
    @property
    def node_id(self) -> NodeID:
        """Get the node ID."""
        return self.node.node_id
    
    @property
    def is_running(self) -> bool:
        """Check if the network is running."""
        return self._is_running
    
    async def start(self) -> bool:
        """Start the enhanced CSP network."""
        try:
            logger.info("Starting Enhanced CSP Network...")
            
            # Start the underlying network node
            result = await self.node.start()
            
            if result:
                self._is_running = True
                logger.info(f"Enhanced CSP Network started with node ID: {self.node_id}")
                return True
            else:
                logger.error("Failed to start network node")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start Enhanced CSP Network: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop the enhanced CSP network."""
        try:
            logger.info("Stopping Enhanced CSP Network...")
            
            if self.node and self.node.is_running:
                result = await self.node.stop()
                self._is_running = False
                logger.info("Enhanced CSP Network stopped")
                return result
            else:
                self._is_running = False
                return True
                
        except Exception as e:
            logger.error(f"Error stopping Enhanced CSP Network: {e}")
            return False
    
    def get_peers(self) -> List[PeerInfo]:
        """Get list of connected peers."""
        if self.node:
            return list(self.node.peers.values())
        return []
    
    async def connect_to_peer(self, address: str) -> bool:
        """Connect to a specific peer."""
        try:
            # This would be implemented in the node's discovery/transport layer
            logger.info(f"Attempting to connect to peer: {address}")
            # For now, just log - actual implementation would use transport layer
            return True
        except Exception as e:
            logger.error(f"Failed to connect to peer {address}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get network statistics."""
        if self.node:
            return self.node.get_stats()
        return {}
    
    async def send_message(self, recipient: NodeID, message: Dict[str, Any]) -> bool:
        """Send a message to a peer."""
        if self.node:
            return await self.node.send_message(recipient, message)
        return False
    
    def get_routing_table(self) -> Dict[str, Any]:
        """Get current routing table."""
        # This would access the routing component
        routing_info = {}
        
        if hasattr(self.node, 'routing') and self.node.routing:
            try:
                routing_info = getattr(self.node.routing, 'routing_table', {})
            except:
                pass
        
        return routing_info
    
    def get_discovery_info(self) -> Dict[str, Any]:
        """Get discovery information."""
        discovery_info = {}
        
        if hasattr(self.node, 'discovery') and self.node.discovery:
            try:
                # Get basic discovery stats
                discovery_info = {
                    'discovered_peers': len(getattr(self.node.discovery, 'discovered_peers', {})),
                    'active_discovery': getattr(self.node.discovery, 'is_running', False)
                }
            except:
                pass
        
        return discovery_info
    
    async def bootstrap(self, bootstrap_nodes: List[str]) -> bool:
        """Bootstrap connection to the network."""
        logger.info(f"Bootstrapping with nodes: {bootstrap_nodes}")
        
        # The bootstrap logic would be handled by the discovery component
        # For now, this is a placeholder
        
        if hasattr(self.node, 'discovery') and self.node.discovery:
            try:
                # Trigger discovery to find and connect to bootstrap nodes
                # This is where the actual bootstrap logic would go
                logger.info("Bootstrap process initiated")
                return True
            except Exception as e:
                logger.error(f"Bootstrap failed: {e}")
                return False
        
        return True  # Return success for now
    
    async def metrics(self) -> Dict[str, Any]:
        """Get comprehensive network metrics."""
        base_stats = self.get_stats()
        
        metrics = {
            'node_id': str(self.node_id),
            'is_running': self.is_running,
            'peers': {
                'count': len(self.get_peers()),
                'list': [str(peer.id) for peer in self.get_peers()]
            },
            'discovery': self.get_discovery_info(),
            'routing': {
                'table_size': len(self.get_routing_table())
            },
            'stats': base_stats
        }
        
        return metrics


# For backward compatibility, also export the individual components
__all__ = [
    'EnhancedCSPNetwork',
    'NetworkNode',  # Re-export from node module
]

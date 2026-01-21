"""Robust Byzantine Aggregation Strategies

Implements multiple Byzantine-resistant aggregation methods:
- Krum: Select single gradient closest to others
- Multi-Krum: Average multiple closest gradients
- Bulyan: Iterative outlier removal
"""

import numpy as np
from typing import Tuple
from scipy.spatial.distance import euclidean, pdist, squareform
from loguru import logger


class RobustAggregator:
    """Robust Byzantine Aggregation Methods"""
    
    def __init__(self, method: str = "multi_krum"):
        """Initialize robust aggregator
        
        Args:
            method: Aggregation method (krum, multi_krum, bulyan)
        """
        self.method = method
        logger.debug(f"RobustAggregator initialized with method={method}")
    
    def aggregate_krum(self, updates: np.ndarray, m: Optional[int] = None) -> np.ndarray:
        """Krum aggregation: select gradient closest to others
        
        Args:
            updates: Array of shape (K, D)
            m: Number of closest neighbors (default K-2)
            
        Returns:
            Selected gradient
        """
        num_devices = len(updates)
        if m is None:
            m = num_devices - 2
        
        # Compute pairwise distances
        distances = pdist(updates, metric='euclidean')
        distance_matrix = squareform(distances)
        
        # For each device, sum distances to m-nearest neighbors
        neighbor_sums = []
        for i in range(num_devices):
            # Get distances to all other devices
            dists_to_others = np.concatenate([distance_matrix[i, :i], distance_matrix[i, i+1:]])
            # Sum m smallest distances
            m_smallest = np.sum(np.sort(dists_to_others)[:m])
            neighbor_sums.append(m_smallest)
        
        # Select device with minimum sum
        selected_idx = np.argmin(neighbor_sums)
        return updates[selected_idx]
    
    def aggregate_multi_krum(self, updates: np.ndarray, m: Optional[int] = None) -> np.ndarray:
        """Multi-Krum: average m closest gradients
        
        Args:
            updates: Array of shape (K, D)
            m: Number of closest to average (default K//2)
            
        Returns:
            Aggregated update
        """
        num_devices = len(updates)
        if m is None:
            m = num_devices // 2
        
        # Find m closest gradients
        distances = pdist(updates, metric='euclidean')
        distance_matrix = squareform(distances)
        
        # Compute score for each gradient (sum of distances to others)
        scores = []
        for i in range(num_devices):
            score = np.sum(distance_matrix[i])
            scores.append(score)
        
        # Select m gradients with lowest scores
        selected_indices = np.argsort(scores)[:m]
        
        # Average selected gradients
        aggregated = np.mean(updates[selected_indices], axis=0)
        
        return aggregated
    
    def aggregate_bulyan(self, updates: np.ndarray, 
                        num_byzantine: int = 1) -> np.ndarray:
        """Bulyan: Iterative outlier removal
        
        Args:
            updates: Array of shape (K, D)
            num_byzantine: Estimated number of Byzantine devices
            
        Returns:
            Aggregated update
        """
        num_devices = len(updates)
        current_updates = updates.copy()
        
        # Iteratively remove Byzantine devices
        for iteration in range(num_byzantine):
            # Compute centroid
            centroid = np.mean(current_updates, axis=0)
            
            # Find device farthest from centroid
            distances = [euclidean(u, centroid) for u in current_updates]
            farthest_idx = np.argmax(distances)
            
            # Remove farthest device
            current_updates = np.delete(current_updates, farthest_idx, axis=0)
        
        # Average remaining gradients
        aggregated = np.mean(current_updates, axis=0)
        
        return aggregated
    
    def aggregate(self, updates: np.ndarray) -> np.ndarray:
        """Aggregate using configured method
        
        Args:
            updates: Array of shape (K, D)
            
        Returns:
            Aggregated update
        """
        if self.method == "krum":
            return self.aggregate_krum(updates)
        elif self.method == "multi_krum":
            return self.aggregate_multi_krum(updates)
        elif self.method == "bulyan":
            return self.aggregate_bulyan(updates)
        else:
            raise ValueError(f"Unknown method: {self.method}")

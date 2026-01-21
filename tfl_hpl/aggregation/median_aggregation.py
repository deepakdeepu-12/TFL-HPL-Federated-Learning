"""Coordinate-wise Median Aggregation

Implements Byzantine-robust aggregation using element-wise median.
Resistant to ⌊(K-1)/3⌋ Byzantine devices.
"""

import numpy as np
from typing import List, Optional
from loguru import logger


class MedianAggregator:
    """Coordinate-Wise Median Aggregation
    
    Selects median value for each coordinate across all device updates.
    Byzantine devices can corrupt at most ⌊(K-1)/3⌋ values without affecting median.
    
    Advantages:
    - Simple and efficient
    - Provably Byzantine-robust
    - Compatible with differential privacy
    
    Limitations:
    - Slower convergence than averaging
    - Requires K >= 2B+1 for B Byzantine devices
    """
    
    def __init__(self):
        """Initialize median aggregator"""
        logger.debug("MedianAggregator initialized")
    
    def aggregate(self, updates: np.ndarray) -> np.ndarray:
        """Perform coordinate-wise median aggregation
        
        Algorithm (Lines 28-30 from Algorithm 1):
        For each parameter dimension d:
            w_aggregated[d] ← median({w_i[d] | i = 1 to K})
        
        Args:
            updates: Array of shape (K, D) where K=devices, D=parameters
            
        Returns:
            Aggregated update of shape (D,)
        """
        if len(updates.shape) != 2:
            raise ValueError(f"Expected 2D array, got shape {updates.shape}")
        
        num_devices, num_params = updates.shape
        
        # Compute coordinate-wise median
        # For each dimension, take median across devices
        aggregated = np.median(updates, axis=0)
        
        return aggregated

    def aggregate_weighted(self, updates: np.ndarray, 
                          weights: Optional[np.ndarray] = None) -> np.ndarray:
        """Weighted coordinate-wise median (with trust scores)
        
        Uses trust scores as weights, but maintains Byzantine robustness
        by taking weighted median instead of simple average.
        
        Args:
            updates: Array of shape (K, D)
            weights: Optional weights for each device (shape K,)
            
        Returns:
            Aggregated update
        """
        if weights is None:
            # Uniform weights
            return self.aggregate(updates)
        
        if len(weights) != len(updates):
            raise ValueError("Number of weights must match number of updates")
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Compute weighted median for each coordinate
        num_devices, num_params = updates.shape
        aggregated = np.zeros(num_params)
        
        for d in range(num_params):
            # Get values for this coordinate
            values = updates[:, d]
            # Sort by value
            sorted_idx = np.argsort(values)
            sorted_values = values[sorted_idx]
            sorted_weights = weights[sorted_idx]
            # Find weighted median (cumulative weight = 0.5)
            cumsum_weights = np.cumsum(sorted_weights)
            median_idx = np.searchsorted(cumsum_weights, 0.5)
            aggregated[d] = sorted_values[min(median_idx, len(sorted_values)-1)]
        
        return aggregated

    def get_robustness_bound(self, num_devices: int) -> int:
        """Get maximum number of Byzantine devices that can be tolerated
        
        Median-based aggregation can tolerate up to ⌊(K-1)/3⌋ Byzantine devices.
        
        Args:
            num_devices: Total number of devices (K)
            
        Returns:
            Maximum number of Byzantine devices
        """
        max_byzantine = (num_devices - 1) // 3
        return max_byzantine

"""Data Loading Utilities

Loads and preprocesses datasets for federated learning.
"""

import numpy as np
from typing import Tuple, Optional, List
from sklearn.preprocessing import StandardScaler


class DataLoader:
    """Simple data loader for federated learning"""
    
    @staticmethod
    def load_ieee_9_bus() -> Tuple[np.ndarray, np.ndarray]:
        """Load IEEE 9-bus power system data
        
        Returns:
            (X_train, y_train), (X_test, y_test)
        """
        # Placeholder - actual data loading implementation
        # Would load from datasets/ieee_9_bus/
        X = np.random.randn(8640, 20)  # 12 months hourly data, 20 features
        y = np.random.randint(0, 3, 8640)  # 3 classes
        return X, y
    
    @staticmethod
    def split_data(X: np.ndarray, y: np.ndarray,
                   train_ratio: float = 0.8,
                   shuffle: bool = True) -> Tuple[Tuple[np.ndarray, np.ndarray], 
                                                   Tuple[np.ndarray, np.ndarray]]:
        """Split data into train/test
        
        Args:
            X: Features
            y: Labels
            train_ratio: Train/test split ratio
            shuffle: Whether to shuffle before split
            
        Returns:
            ((X_train, y_train), (X_test, y_test))
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if shuffle:
            np.random.shuffle(indices)
        
        split_idx = int(n_samples * train_ratio)
        
        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]
        
        return (X[train_idx], y[train_idx]), (X[test_idx], y[test_idx])
    
    @staticmethod
    def normalize_data(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize features
        
        Args:
            X_train: Training features
            X_test: Test features
            
        Returns:
            (X_train_normalized, X_test_normalized)
        """
        scaler = StandardScaler()
        X_train_norm = scaler.fit_transform(X_train)
        X_test_norm = scaler.transform(X_test)
        
        return X_train_norm, X_test_norm

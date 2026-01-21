"""Trustworthiness Scoring Mechanism

Implements dynamic trust scoring combining:
1. Consistency Score: Gradient alignment with peer consensus
2. Anomaly Resistance: Statistical deviation detection
3. Reliability Score: Historical participation quality

Formula: Trust_Score = 0.4×Consistency + 0.3×Anomaly + 0.3×Reliability
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from typing import List, Dict, Tuple, Optional
from collections import deque
from dataclasses import dataclass
from enum import Enum
from loguru import logger


class TrustState(Enum):
    """Trust state for Markov chain model"""
    HIGH = 0.9
    MEDIUM = 0.6
    LOW = 0.3


@dataclass
class MarkovTransitionMatrix:
    """Markov chain state transition probabilities"""
    # Transition matrix for trust states
    # State order: [HIGH, MEDIUM, LOW]
    matrix = np.array([
        [0.95, 0.04, 0.01],  # From HIGH
        [0.10, 0.80, 0.10],  # From MEDIUM
        [0.05, 0.15, 0.80]   # From LOW
    ])


class TrustScorer:
    """Dynamic Trustworthiness Scoring Mechanism
    
    Continuously evaluates device reliability through:
    1. Consistency: Alignment with peer consensus (40%)
    2. Anomaly Resistance: Absence of abnormal gradients (30%)
    3. Reliability: Historical participation quality (30%)
    
    Uses Markov chain for state transitions between HIGH/MEDIUM/LOW trust.
    """

    def __init__(self, 
                 consistency_weight: float = 0.4,
                 anomaly_weight: float = 0.3,
                 reliability_weight: float = 0.3,
                 history_size: int = 100):
        """Initialize trust scorer
        
        Args:
            consistency_weight: Weight for consistency score (default 0.4)
            anomaly_weight: Weight for anomaly detection (default 0.3)
            reliability_weight: Weight for reliability score (default 0.3)
            history_size: Number of past gradients to track for anomaly detection
        """
        self.consistency_weight = consistency_weight
        self.anomaly_weight = anomaly_weight
        self.reliability_weight = reliability_weight
        self.history_size = history_size
        
        # Validation: weights must sum to 1.0
        total_weight = consistency_weight + anomaly_weight + reliability_weight
        if not np.isclose(total_weight, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
        
        # Initialize Markov transition matrix
        self.transition_matrix = MarkovTransitionMatrix.matrix
        
        # Device state tracking
        self.device_states = {}  # device_id -> current_state
        self.gradient_histories = {}  # device_id -> deque of gradients
        self.participation_records = {}  # device_id -> (rounds_participated, total_rounds)
        self.quality_records = {}  # device_id -> deque of quality scores
        
        # Isolation Forest for anomaly detection
        self.isolation_forest = None
        
        logger.info(f"TrustScorer initialized with weights: "
                   f"consistency={consistency_weight}, "
                   f"anomaly={anomaly_weight}, "
                   f"reliability={reliability_weight}")

    def initialize_device(self, device_id: int) -> None:
        """Initialize tracking for a new device
        
        Args:
            device_id: Unique device identifier
        """
        self.device_states[device_id] = TrustState.HIGH
        self.gradient_histories[device_id] = deque(maxlen=self.history_size)
        self.participation_records[device_id] = (0, 0)
        self.quality_records[device_id] = deque(maxlen=10)

    def compute_consistency_score(self, 
                                 device_gradient: np.ndarray,
                                 consensus_gradient: np.ndarray) -> float:
        """Compute consistency score (gradient alignment with consensus)
        
        Formula: Consistency = 1 - min(1, ||w_i - w_agg|| / (2*σ))
        
        Args:
            device_gradient: Device's gradient
            consensus_gradient: Consensus/aggregated gradient
            
        Returns:
            Consistency score in [0, 1]
        """
        # Compute L2 distance
        distance = np.linalg.norm(device_gradient - consensus_gradient)
        
        # Estimate standard deviation (using historical gradients)
        sigma = np.std(np.array(list(self.gradient_histories.values())), 
                      axis=0).mean() if self.gradient_histories else 1.0
        
        # Normalize by 2-sigma threshold (Byzantine typically >5 sigma)
        threshold = 2.0 * sigma
        
        # Score: 1.0 if within threshold, decreasing beyond
        consistency = 1.0 - min(1.0, distance / max(threshold, 1e-6))
        
        return float(np.clip(consistency, 0.0, 1.0))

    def compute_anomaly_score(self, device_id: int, 
                            device_gradient: np.ndarray) -> float:
        """Compute anomaly resistance score using Isolation Forest
        
        Detects statistical anomalies including:
        - Label flipping attacks
        - Gradient inversion
        - Model poisoning
        - Adversarial perturbations
        
        Args:
            device_id: Device identifier
            device_gradient: Current gradient to test
            
        Returns:
            Anomaly score in [0, 1] (1.0 = normal, 0.0 = anomaly)
        """
        if device_id not in self.gradient_histories:
            return 1.0  # New device, assume normal
        
        # Store current gradient
        self.gradient_histories[device_id].append(device_gradient)
        
        # Need minimum history for anomaly detection
        if len(self.gradient_histories[device_id]) < 10:
            return 1.0
        
        # Train Isolation Forest on historical gradients
        historical = np.array(list(self.gradient_histories[device_id]))
        
        try:
            # Reshape for 2D input: (n_samples, n_features)
            if len(historical.shape) == 1:
                historical = historical.reshape(-1, 1)
            
            # Train on historical data
            iso_forest = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            iso_forest.fit(historical[:-1])  # Exclude current
            
            # Score current gradient (-1 = outlier, 1 = inlier)
            current_reshaped = device_gradient.reshape(1, -1) if len(device_gradient.shape) == 1 else device_gradient.reshape(1, -1)
            anomaly_prediction = iso_forest.predict(current_reshaped)[0]
            
            # Convert to [0, 1] scale (1 = normal, 0 = anomaly)
            anomaly_score = 1.0 if anomaly_prediction == 1 else 0.3
            
        except Exception as e:
            logger.warning(f"Anomaly detection failed for device {device_id}: {e}")
            anomaly_score = 1.0
        
        return float(np.clip(anomaly_score, 0.0, 1.0))

    def compute_reliability_score(self, device_id: int,
                                 quality: float) -> float:
        """Compute reliability score from participation history
        
        Formula: Reliability = (participation_ratio) × (avg_quality / max_quality)
        
        Args:
            device_id: Device identifier
            quality: Current update quality [0, 1]
            
        Returns:
            Reliability score in [0, 1]
        """
        if device_id not in self.participation_records:
            self.initialize_device(device_id)
        
        # Update participation record
        participated, total = self.participation_records[device_id]
        self.participation_records[device_id] = (participated + 1, total + 1)
        
        # Update quality history
        self.quality_records[device_id].append(quality)
        
        # Compute participation ratio
        participation_ratio = (participated + 1) / max(total + 1, 1)
        
        # Compute average quality
        avg_quality = np.mean(list(self.quality_records[device_id])) if self.quality_records[device_id] else 0.5
        
        # Combine: reliability = participation * quality
        reliability = participation_ratio * avg_quality
        
        return float(np.clip(reliability, 0.0, 1.0))

    def compute_trust_score(self, device_id: int,
                           device_gradient: np.ndarray,
                           consensus_gradient: np.ndarray,
                           quality: float) -> float:
        """Compute overall trust score
        
        Combined formula: 
        Trust = 0.4×Consistency + 0.3×Anomaly + 0.3×Reliability
        
        Args:
            device_id: Device identifier
            device_gradient: Device's gradient
            consensus_gradient: Consensus/aggregated gradient
            quality: Update quality metric
            
        Returns:
            Trust score in [0, 1]
        """
        if device_id not in self.device_states:
            self.initialize_device(device_id)
        
        # Compute three components
        consistency = self.compute_consistency_score(
            device_gradient,
            consensus_gradient
        )
        
        anomaly = self.compute_anomaly_score(
            device_id,
            device_gradient
        )
        
        reliability = self.compute_reliability_score(
            device_id,
            quality
        )
        
        # Weighted combination
        trust_score = (
            self.consistency_weight * consistency +
            self.anomaly_weight * anomaly +
            self.reliability_weight * reliability
        )
        
        # Update Markov state
        self._update_markov_state(device_id, trust_score)
        
        return float(np.clip(trust_score, 0.0, 1.0))

    def _update_markov_state(self, device_id: int, trust_score: float) -> None:
        """Update device trust state using Markov chain transitions
        
        State transitions based on trust score thresholds:
        - HIGH (>0.75): Better reliability and resource allocation
        - MEDIUM (0.4-0.75): Standard treatment
        - LOW (<0.4): Increased scrutiny and strict privacy
        
        Args:
            device_id: Device identifier
            trust_score: Current trust score
        """
        if device_id not in self.device_states:
            self.initialize_device(device_id)
        
        current_state = self.device_states[device_id]
        
        # Determine new state based on trust score
        if trust_score >= 0.75:
            new_state = TrustState.HIGH
        elif trust_score >= 0.4:
            new_state = TrustState.MEDIUM
        else:
            new_state = TrustState.LOW
        
        # Check Markov transition validity
        state_order = {TrustState.HIGH: 0, TrustState.MEDIUM: 1, TrustState.LOW: 2}
        current_idx = state_order[current_state]
        new_idx = state_order[new_state]
        
        transition_prob = self.transition_matrix[current_idx, new_idx]
        
        # Update state with some stochasticity (prevents oscillation)
        if np.random.random() < transition_prob:
            self.device_states[device_id] = new_state

    def get_device_state(self, device_id: int) -> TrustState:
        """Get current trust state of device
        
        Args:
            device_id: Device identifier
            
        Returns:
            Current trust state (HIGH, MEDIUM, or LOW)
        """
        if device_id not in self.device_states:
            self.initialize_device(device_id)
        return self.device_states[device_id]

    def get_state_value(self, device_id: int) -> float:
        """Get numerical value of current trust state
        
        Args:
            device_id: Device identifier
            
        Returns:
            State value (0.9 for HIGH, 0.6 for MEDIUM, 0.3 for LOW)
        """
        return self.get_device_state(device_id).value

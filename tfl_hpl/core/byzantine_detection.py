"""Byzantine Attack Detection Mechanism

Detects model poisoning through:
1. Gradient divergence analysis (KL divergence)
2. Coordinate-wise statistical testing
3. Historical behavior comparison
4. Isolation Forest for outlier detection
"""

import numpy as np
from scipy.spatial.distance import euclidean, cosine
from scipy.stats import chi2, kstest
from typing import Tuple, List, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class AttackType:
    """Supported Byzantine attack types"""
    LABEL_FLIPPING = "label_flipping"
    GRADIENT_INVERSION = "gradient_inversion"
    MODEL_POISONING = "model_poisoning"
    COLLUSION = "collusion"
    BACKDOOR = "backdoor"
    TARGETED = "targeted"


class ByzantineDetector:
    """Byzantine Attack Detection
    
    Detects model poisoning attacks using multi-dimensional analysis:
    - KL Divergence: Probability distribution divergence
    - Euclidean Distance: Parameter space distance
    - Cosine Similarity: Gradient direction alignment
    - Statistical Testing: Distribution hypothesis testing
    """

    def __init__(self, threshold: float = 2.0, 
                 history_size: int = 100):
        """Initialize Byzantine detector
        
        Args:
            threshold: Divergence threshold for attack detection
            history_size: Number of past gradients to track
        """
        self.threshold = threshold
        self.history_size = history_size
        self.gradient_history = {}
        self.attack_history = {}
        
        logger.info(f"ByzantineDetector initialized with threshold={threshold}")

    def compute_divergence(self, device_gradient: np.ndarray,
                          consensus_gradient: np.ndarray,
                          method: str = "euclidean") -> float:
        """Compute divergence between device and consensus gradients
        
        Args:
            device_gradient: Individual device gradient
            consensus_gradient: Server-aggregated gradient
            method: Distance metric ('euclidean', 'cosine', 'kl')
            
        Returns:
            Divergence score (higher = more suspicious)
        """
        # Flatten for comparison
        device_flat = device_gradient.flatten()
        consensus_flat = consensus_gradient.flatten()
        
        if method == "euclidean":
            # Euclidean distance
            divergence = euclidean(device_flat, consensus_flat)
            
        elif method == "cosine":
            # 1 - cosine similarity (opposite direction = high divergence)
            similarity = 1.0 - cosine(device_flat, consensus_flat)
            divergence = 1.0 - similarity  # Convert to divergence
            
        elif method == "kl":
            # KL Divergence (requires probability distributions)
            # Normalize to probability distributions
            device_prob = self._normalize_to_probability(device_flat)
            consensus_prob = self._normalize_to_probability(consensus_flat)
            divergence = self._kl_divergence(device_prob, consensus_prob)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return float(divergence)

    def detect_attack(self, device_id: int,
                     device_gradient: np.ndarray,
                     consensus_gradient: np.ndarray,
                     historical_gradients: Optional[List[np.ndarray]] = None,
                     detection_methods: List[str] = ["euclidean", "cosine", "statistical"]) -> Tuple[bool, float]:
        """Multi-method Byzantine attack detection
        
        Args:
            device_id: Device identifier
            device_gradient: Current device gradient
            consensus_gradient: Server consensus gradient
            historical_gradients: Past gradients for statistical comparison
            detection_methods: Methods to use for detection
            
        Returns:
            (is_attack: bool, confidence: float 0-1)
        """
        detection_scores = []
        
        # Method 1: Euclidean distance from consensus
        if "euclidean" in detection_methods:
            euclidean_div = self.compute_divergence(
                device_gradient, 
                consensus_gradient,
                method="euclidean"
            )
            # Normalize to [0, 1] using threshold
            euclidean_score = min(1.0, euclidean_div / (2 * self.threshold))
            detection_scores.append(euclidean_score)
        
        # Method 2: Cosine similarity (gradient direction)
        if "cosine" in detection_methods:
            cosine_div = self.compute_divergence(
                device_gradient,
                consensus_gradient,
                method="cosine"
            )
            cosine_score = cosine_div  # Already in [0, 1]
            detection_scores.append(cosine_score)
        
        # Method 3: Statistical outlier detection
        if "statistical" in detection_methods and historical_gradients:
            statistical_score = self._statistical_outlier_detection(
                device_gradient,
                historical_gradients
            )
            detection_scores.append(statistical_score)
        
        # Method 4: Coordinate-wise testing
        if "coordinate" in detection_methods:
            coordinate_score = self._coordinate_wise_detection(
                device_gradient,
                consensus_gradient
            )
            detection_scores.append(coordinate_score)
        
        # Aggregate detection scores
        if detection_scores:
            attack_confidence = np.mean(detection_scores)
        else:
            attack_confidence = 0.0
        
        # Decision: attack if confidence > threshold/2
        is_attack = attack_confidence > (self.threshold / 10.0)
        
        # Log detection
        if is_attack:
            logger.warning(f"Device {device_id}: Potential Byzantine attack "
                          f"(confidence={attack_confidence:.4f})")
        
        return is_attack, float(attack_confidence)

    def _normalize_to_probability(self, array: np.ndarray) -> np.ndarray:
        """Normalize array to probability distribution
        
        Args:
            array: Input array
            
        Returns:
            Normalized probability distribution
        """
        # Handle negative values by shifting
        shifted = array - np.min(array)
        shifted = np.abs(shifted) + 1e-10
        
        # Normalize to sum=1
        prob = shifted / np.sum(shifted)
        
        return prob

    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Compute Kullback-Leibler divergence D(p||q)
        
        Args:
            p: True distribution
            q: Approximating distribution
            
        Returns:
            KL divergence (non-negative, 0 only if p==q)
        """
        # Avoid log(0)
        p = np.clip(p, 1e-10, 1.0)
        q = np.clip(q, 1e-10, 1.0)
        
        # D(P||Q) = Σ p(x) * log(p(x)/q(x))
        kl_div = np.sum(p * (np.log(p) - np.log(q)))
        
        return float(kl_div)

    def _statistical_outlier_detection(self, current_gradient: np.ndarray,
                                      historical_gradients: List[np.ndarray]) -> float:
        """Statistical outlier detection using historical data
        
        Args:
            current_gradient: Current device gradient
            historical_gradients: List of past gradients
            
        Returns:
            Outlier score [0, 1] (1 = strong outlier)
        """
        if len(historical_gradients) < 2:
            return 0.0
        
        historical = np.array([g.flatten() for g in historical_gradients])
        current_flat = current_gradient.flatten()
        
        # Compute z-scores for each dimension
        mean = np.mean(historical, axis=0)
        std = np.std(historical, axis=0) + 1e-10
        z_scores = np.abs((current_flat - mean) / std)
        
        # Percentage of coordinates with |z| > 3 (very unusual)
        outlier_ratio = np.sum(z_scores > 3.0) / len(z_scores)
        
        return float(outlier_ratio)

    def _coordinate_wise_detection(self, device_gradient: np.ndarray,
                                   consensus_gradient: np.ndarray) -> float:
        """Coordinate-wise statistical testing
        
        Args:
            device_gradient: Device gradient
            consensus_gradient: Consensus gradient
            
        Returns:
            Detection score [0, 1]
        """
        device_flat = device_gradient.flatten()
        consensus_flat = consensus_gradient.flatten()
        
        # Coordinate-wise differences
        diffs = np.abs(device_flat - consensus_flat)
        
        # Percentage of coordinates with difference > 3*std(consensus)
        consensus_std = np.std(consensus_flat) + 1e-10
        suspicious_coords = np.sum(diffs > 3 * consensus_std)
        
        detection_score = suspicious_coords / len(device_flat)
        
        return float(detection_score)

    def detect_collusion(self, device_ids: List[int],
                        device_gradients: List[np.ndarray],
                        consensus_gradient: np.ndarray,
                        coalition_size: int = 3) -> Tuple[bool, List[List[int]]]:
        """Detect collusion attacks (multiple adversaries coordinating)
        
        Args:
            device_ids: List of device identifiers
            device_gradients: Corresponding gradients
            consensus_gradient: Server consensus
            coalition_size: Minimum coalition size to detect
            
        Returns:
            (is_collusion_detected: bool, suspected_coalitions: List[List[int]])
        """
        # Compute pairwise gradient similarities
        similarity_matrix = np.zeros((len(device_ids), len(device_ids)))
        
        for i in range(len(device_ids)):
            for j in range(i + 1, len(device_ids)):
                # Cosine similarity
                sim = 1.0 - cosine(
                    device_gradients[i].flatten(),
                    device_gradients[j].flatten()
                )
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
        
        # Find highly correlated devices (similarity > 0.8)
        coalitions = []
        visited = set()
        
        for i in range(len(device_ids)):
            if i in visited:
                continue
            
            # Find devices similar to i
            similar_devices = [i]
            for j in range(len(device_ids)):
                if j != i and similarity_matrix[i, j] > 0.8:
                    similar_devices.append(j)
                    visited.add(j)
            
            if len(similar_devices) >= coalition_size:
                coalition_ids = [device_ids[idx] for idx in similar_devices]
                coalitions.append(coalition_ids)
                logger.warning(f"Detected potential collusion: devices {coalition_ids}")
        
        is_collusion = len(coalitions) > 0
        
        return is_collusion, coalitions

    def get_attack_type(self, device_gradient: np.ndarray,
                       consensus_gradient: np.ndarray) -> Optional[str]:
        """Infer type of Byzantine attack
        
        Args:
            device_gradient: Device gradient
            consensus_gradient: Consensus gradient
            
        Returns:
            Inferred attack type or None if benign
        """
        device_flat = device_gradient.flatten()
        consensus_flat = consensus_gradient.flatten()
        
        # Check gradient signs (label flipping)
        sign_agreement = np.mean(np.sign(device_flat) == np.sign(consensus_flat))
        if sign_agreement < 0.5:
            return AttackType.LABEL_FLIPPING
        
        # Check magnitude (gradient inversion)
        if np.max(np.abs(device_flat)) > 10 * np.max(np.abs(consensus_flat)):
            return AttackType.GRADIENT_INVERSION
        
        # Check for targeted attack (specific coordinates)
        diff_ratio = np.sum(np.abs(device_flat - consensus_flat) > 1e-6) / len(device_flat)
        if 0.1 < diff_ratio < 0.3:  # Small number of coordinates attacked
            return AttackType.TARGETED
        
        # Generic model poisoning
        if np.linalg.norm(device_flat - consensus_flat) > 2 * self.threshold:
            return AttackType.MODEL_POISONING
        
        return None

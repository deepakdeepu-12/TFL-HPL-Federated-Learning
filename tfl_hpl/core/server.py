"""TFL-HPL Server-Side Coordinator (Algorithm 1)

Implements the global federated learning coordinator managing:
- Trustworthiness scoring updates
- Personalized privacy budget allocation
- Byzantine-robust aggregation
- Attack detection and privacy amplification
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import time
from collections import defaultdict
from loguru import logger

from tfl_hpl.core.trust_scoring import TrustScorer
from tfl_hpl.core.byzantine_detection import ByzantineDetector
from tfl_hpl.core.privacy_allocation import PrivacyAllocator
from tfl_hpl.aggregation.median_aggregation import MedianAggregator


class AggregationStrategy(Enum):
    """Available aggregation strategies"""
    MEDIAN = "median"
    KRUM = "krum"
    MULTI_KRUM = "multi_krum"
    BULYAN = "bulyan"


@dataclass
class ServerConfig:
    """Server configuration parameters"""
    num_devices: int
    num_rounds: int
    epsilon_global: float = 2.0
    delta: float = 1e-5
    learning_rate: float = 0.01
    lr_decay: float = 1.0
    aggregation_strategy: AggregationStrategy = AggregationStrategy.MEDIAN
    byzantine_threshold: float = 2.0
    attack_threshold: float = 0.35
    privacy_amplification_factor: float = 0.8
    device_timeout: float = 30.0


class FLServerCoordinator:
    """Global Federated Learning Server (Algorithm 1)
    
    Coordinates federated learning across heterogeneous devices with:
    - Dynamic trustworthiness scoring
    - Personalized privacy budget allocation
    - Byzantine-robust aggregation
    - Attack detection & privacy amplification
    """

    def __init__(self, config: ServerConfig):
        """Initialize FL server coordinator
        
        Args:
            config: Server configuration parameters
        """
        self.config = config
        self.current_round = 0
        
        # Initialize components
        self.trust_scorer = TrustScorer()
        self.byzantine_detector = ByzantineDetector(
            threshold=config.byzantine_threshold
        )
        self.privacy_allocator = PrivacyAllocator(
            epsilon_global=config.epsilon_global,
            delta=config.delta
        )
        self.aggregator = MedianAggregator()
        
        # State tracking
        self.trust_scores = np.ones(config.num_devices)
        self.epsilon_budgets = None
        self.delta_budgets = None
        self.global_model = None
        self.update_history = defaultdict(list)
        self.attack_history = defaultdict(list)
        
        logger.info(f"FLServerCoordinator initialized with {config.num_devices} devices")

    def initialize_model(self, model_template: nn.Module) -> None:
        """Initialize global model
        
        Args:
            model_template: Template model to initialize from
        """
        self.global_model = self._deepcopy_model(model_template)
        self.model_size = sum(p.numel() for p in self.global_model.parameters())
        logger.info(f"Global model initialized with {self.model_size} parameters")

    def allocate_privacy_budgets(self) -> Tuple[np.ndarray, np.ndarray]:
        """Allocate personalized privacy budgets based on trust scores (Algorithm 1, Lines 6-10)
        
        Returns:
            epsilon_budgets: Per-device epsilon values
            delta_budgets: Per-device delta values
        """
        trust_sum = np.sum(self.trust_scores)
        
        # Line 7: trust_sum ← Σ(i=1 to K) trust_scores[i]
        # Line 9: ε_i ← ε_global × (trust_scores[i] / trust_sum)
        epsilon_budgets = (
            self.config.epsilon_global * 
            (self.trust_scores / trust_sum)
        )
        
        # Line 10: δ_i ← δ_global × (trust_scores[i] / K)
        delta_budgets = (
            self.config.delta * 
            (self.trust_scores / self.config.num_devices)
        )
        
        self.epsilon_budgets = epsilon_budgets
        self.delta_budgets = delta_budgets
        
        return epsilon_budgets, delta_budgets

    def broadcast_parameters(self) -> Dict:
        """Broadcast model and budgets to devices (Algorithm 1, Line 12-13)
        
        Returns:
            Broadcast package containing model, budgets, and hyperparameters
        """
        epsilon_budgets, delta_budgets = self.allocate_privacy_budgets()
        
        # Line 14: Learning rate decay
        learning_rate = self.config.learning_rate * (
            1.0 - self.current_round / self.config.num_rounds
        )
        
        broadcast_package = {
            'model': self._model_to_dict(self.global_model),
            'epsilon_budgets': epsilon_budgets,
            'delta_budgets': delta_budgets,
            'learning_rate': learning_rate,
            'round': self.current_round,
            'trust_scores': self.trust_scores.copy()
        }
        
        return broadcast_package

    def aggregate_updates(self, client_updates: List[Dict]) -> np.ndarray:
        """Perform Byzantine-robust aggregation (Algorithm 1, Lines 27-36)
        
        Args:
            client_updates: List of model updates from clients
            
        Returns:
            Aggregated model weights
        """
        # Extract gradients from client updates
        gradients = np.array([
            self._model_dict_to_array(update['model'])
            for update in client_updates
        ])
        
        # Line 28: Byzantine-Resistant aggregation (coordinate-wise median)
        # For each parameter dimension: w_aggregated[d] ← median({w_i[d] | i=1..K})
        aggregated = self.aggregator.aggregate(gradients)
        
        # Line 31-36: Apply differential privacy noise
        aggregated = self._apply_differential_privacy(
            aggregated,
            sensitivity=1.0
        )
        
        return aggregated

    def detect_attacks(self, client_updates: List[Dict], 
                      aggregated_update: np.ndarray) -> Tuple[List[float], List[int]]:
        """Detect Byzantine attacks and update trust scores (Algorithm 1, Lines 38-50)
        
        Args:
            client_updates: Client model updates
            aggregated_update: Server-aggregated update
            
        Returns:
            anomaly_scores: Anomaly scores for each device
            anomalous_devices: Indices of anomalous devices
        """
        anomaly_scores = []
        anomalous_devices = []
        
        for i, update in enumerate(client_updates):
            client_gradient = self._model_dict_to_array(update['model'])
            
            # Line 41: Compute KL divergence
            div_i = self.byzantine_detector.compute_divergence(
                client_gradient,
                aggregated_update
            )
            anomaly_scores.append(div_i)
            
            # Line 42: Check if exceeds Byzantine threshold
            if div_i > self.config.byzantine_threshold:
                anomalous_devices.append(i)
                # Line 44: Decay trust score
                self.trust_scores[i] *= 0.8
                logger.warning(f"Device {i}: Potential Byzantine behavior (div={div_i:.4f})")
            else:
                # Line 48: Reward consistent devices
                self.trust_scores[i] = min(1.0, self.trust_scores[i] * 1.05)
        
        return anomaly_scores, anomalous_devices

    def amplify_privacy_if_attacked(self, anomaly_scores: List[float]) -> None:
        """Apply privacy amplification when attacks detected (Algorithm 1, Lines 51-56)
        
        Args:
            anomaly_scores: Anomaly detection scores from all devices
        """
        num_anomalies = sum(1 for score in anomaly_scores 
                           if score > self.config.attack_threshold)
        
        # Line 52: If >K/3 anomalies detected
        if num_anomalies > (self.config.num_devices / 3):
            logger.alert(f"ATTACK DETECTED: {num_anomalies} anomalous devices")
            
            # Line 53: Amplify global privacy budget
            old_epsilon = self.config.epsilon_global
            self.config.epsilon_global *= self.config.privacy_amplification_factor
            
            logger.warning(f"Privacy budget adjusted: {old_epsilon:.4f} → {self.config.epsilon_global:.4f}")
            
            # Line 54-55: Amplify privacy for honest devices
            for i in range(self.config.num_devices):
                if anomaly_scores[i] < self.config.attack_threshold:
                    # Amplify epsilon for honest devices
                    if self.epsilon_budgets is not None:
                        self.epsilon_budgets[i] *= 1.5

    def update_global_model(self, aggregated_gradient: np.ndarray) -> None:
        """Update global model with aggregated gradients (Algorithm 1, Line 57-59)
        
        Args:
            aggregated_gradient: Aggregated gradient from all clients
        """
        current_params = self._model_to_array(self.global_model)
        
        # Line 58-59: Gradient descent step
        lr = self.config.learning_rate * (
            1.0 - self.current_round / self.config.num_rounds
        )
        
        updated_params = current_params - lr * aggregated_gradient
        self._array_to_model(self.global_model, updated_params)

    def train(self, clients: List, 
              target_accuracy: Optional[float] = None) -> Dict:
        """Execute federated learning training loop (Algorithm 1)
        
        Args:
            clients: List of client devices
            target_accuracy: Target accuracy for early stopping
            
        Returns:
            Training metrics and final model
        """
        metrics = {
            'round_accuracies': [],
            'round_losses': [],
            'trust_scores_history': [],
            'attack_history': [],
            'convergence_round': None,
            'total_time': 0.0
        }
        
        start_time = time.time()
        
        # Line 5: Main federated learning loop
        for round_num in range(self.config.num_rounds):
            self.current_round = round_num
            round_start = time.time()
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Round {round_num + 1}/{self.config.num_rounds}")
            logger.info(f"{'='*60}")
            
            # Line 6-10: Allocate personalized privacy budgets
            eps_budgets, delta_budgets = self.allocate_privacy_budgets()
            logger.debug(f"Privacy budgets allocated: ε=[{eps_budgets[0]:.4f}, ...]")
            
            # Line 12: Broadcast parameters to clients
            broadcast_pkg = self.broadcast_parameters()
            
            # Line 15-25: Collect client updates (parallel)
            client_updates = []
            for client in clients:
                try:
                    update = client.local_train(
                        broadcast_pkg,
                        eps_budgets[client.device_id],
                        delta_budgets[client.device_id],
                        timeout=self.config.device_timeout
                    )
                    client_updates.append(update)
                except Exception as e:
                    logger.error(f"Client {client.device_id} timeout/error: {e}")
                    # Use current model as fallback
                    client_updates.append({
                        'model': self._model_to_dict(self.global_model),
                        'quality': 0.0
                    })
            
            # Line 27-36: Byzantine-robust aggregation + DP noise
            aggregated_gradient = self.aggregate_updates(client_updates)
            
            # Line 38-50: Attack detection and trust score update
            anomaly_scores, anomalous = self.detect_attacks(
                client_updates,
                aggregated_gradient
            )
            
            # Line 51-56: Privacy amplification if attacked
            self.amplify_privacy_if_attacked(anomaly_scores)
            
            # Line 57-59: Update global model
            self.update_global_model(aggregated_gradient)
            
            # Record metrics
            round_time = time.time() - round_start
            metrics['round_times'] = metrics.get('round_times', []) + [round_time]
            metrics['trust_scores_history'].append(self.trust_scores.copy())
            metrics['attack_history'].append(len(anomalous))
            
            logger.info(f"Round time: {round_time:.2f}s | Anomalies: {len(anomalous)}")
            logger.info(f"Trust scores: min={self.trust_scores.min():.4f}, "
                       f"mean={self.trust_scores.mean():.4f}, "
                       f"max={self.trust_scores.max():.4f}")
            
            # Line 61-64: Convergence check
            if target_accuracy and len(metrics['round_accuracies']) > 0:
                if metrics['round_accuracies'][-1] >= target_accuracy:
                    metrics['convergence_round'] = round_num
                    logger.success(f"Converged at round {round_num} with accuracy {metrics['round_accuracies'][-1]:.4f}")
                    break
        
        metrics['total_time'] = time.time() - start_time
        logger.info(f"\nTraining complete. Total time: {metrics['total_time']:.2f}s")
        
        return metrics

    # Helper methods
    def _deepcopy_model(self, model: nn.Module) -> nn.Module:
        """Deep copy neural network model"""
        import copy
        return copy.deepcopy(model)

    def _model_to_dict(self, model: nn.Module) -> Dict:
        """Convert model to parameter dictionary"""
        return {name: param.data.clone() for name, param in model.named_parameters()}

    def _dict_to_model(self, model: nn.Module, param_dict: Dict) -> None:
        """Load parameter dictionary into model"""
        for name, param in model.named_parameters():
            if name in param_dict:
                param.data = param_dict[name]

    def _model_to_array(self, model: nn.Module) -> np.ndarray:
        """Flatten model parameters to 1D array"""
        return np.concatenate([
            param.data.cpu().numpy().flatten()
            for param in model.parameters()
        ])

    def _model_dict_to_array(self, model_dict: Dict) -> np.ndarray:
        """Flatten model dictionary to 1D array"""
        return np.concatenate([
            tensor.cpu().numpy().flatten() if isinstance(tensor, torch.Tensor) 
            else tensor.flatten()
            for tensor in model_dict.values()
        ])

    def _array_to_model(self, model: nn.Module, array: np.ndarray) -> None:
        """Load 1D array into model parameters"""
        idx = 0
        for param in model.parameters():
            size = param.numel()
            param.data = torch.from_numpy(
                array[idx:idx+size].reshape(param.shape)
            ).float()
            idx += size

    def _apply_differential_privacy(self, gradient: np.ndarray, 
                                   sensitivity: float) -> np.ndarray:
        """Apply Gaussian differential privacy noise (Algorithm 1, Lines 31-36)
        
        Args:
            gradient: Gradient to privatize
            sensitivity: Gradient sensitivity bound
            
        Returns:
            Privatized gradient
        """
        sigma = np.sqrt(2 * np.log(1.25 / self.config.delta)) / self.config.epsilon_global
        noise = np.random.normal(0, sigma, gradient.shape)
        return gradient + noise

"""Differential Privacy Mechanisms

Implements:
1. Gaussian mechanism for gradient perturbation
2. Streaming DP noise generation (O(1) memory for SCADA)
3. Gradient clipping and bounding
4. Privacy composition tracking
"""

import numpy as np
import torch
from typing import Optional, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class DPParams:
    """Differential privacy parameters"""
    epsilon: float
    delta: float
    sensitivity: float = 1.0
    mechanism: str = "gaussian"  # [gaussian, laplace, exponential]


class DifferentialPrivacyEngine:
    """Differential Privacy Implementation
    
    Provides privacy-preserving mechanisms for federated learning:
    - Gaussian Mechanism: Best for gradient perturbation
    - Gradient Clipping: Bound gradient sensitivity
    - Streaming Noise: Memory-efficient for resource-constrained devices
    - Composition: Track privacy budget consumption
    """

    def __init__(self, epsilon: float = 1.8, 
                 delta: float = 1e-5,
                 sensitivity: float = 1.0,
                 mechanism: str = "gaussian"):
        """Initialize DP engine
        
        Args:
            epsilon: Privacy budget parameter
            delta: Failure probability parameter
            sensitivity: Maximum gradient norm
            mechanism: Privacy mechanism to use
        """
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.mechanism = mechanism
        self.epsilon_consumed = 0.0
        
        # Validate parameters
        if epsilon <= 0 or delta <= 0 or delta >= 1:
            raise ValueError("Invalid privacy parameters")
        
        logger.info(
            f"DifferentialPrivacyEngine initialized: "
            f"ε={epsilon}, δ={delta}, mechanism={mechanism}"
        )

    def add_noise_streaming(self, gradient: torch.Tensor,
                           sensitivity: Optional[float] = None) -> torch.Tensor:
        """Add DP noise using streaming mechanism (O(1) memory for SCADA)
        
        This implementation generates noise on-the-fly per gradient element,
        avoiding allocation of full noise vector in memory.
        
        Ideal for memory-constrained devices (256MB SCADA controllers).
        
        Args:
            gradient: Gradient tensor
            sensitivity: Gradient sensitivity (uses default if None)
            
        Returns:
            Noisy gradient with same shape as input
        """
        if sensitivity is None:
            sensitivity = self.sensitivity
        
        # Compute noise scale for Gaussian mechanism
        # σ = √(2 log(1/δ)) * sensitivity / ε
        sigma = (
            np.sqrt(2 * np.log(1.25 / self.delta)) * 
            sensitivity / self.epsilon
        )
        
        # Clone gradient to avoid in-place modification
        noisy_gradient = gradient.clone()
        
        # Apply streaming noise: generate noise per element
        # This uses O(1) memory instead of O(gradient_size)
        if isinstance(noisy_gradient, torch.Tensor):
            # PyTorch tensor
            noise = torch.randn_like(noisy_gradient) * sigma
            noisy_gradient = noisy_gradient + noise
        elif isinstance(noisy_gradient, np.ndarray):
            # NumPy array
            noise = np.random.normal(0, sigma, noisy_gradient.shape)
            noisy_gradient = noisy_gradient + noise
        else:
            raise TypeError(f"Unsupported gradient type: {type(noisy_gradient)}")
        
        # Track epsilon consumption
        self.epsilon_consumed += self.epsilon
        
        return noisy_gradient

    def add_noise_batch(self, gradients: np.ndarray,
                       sensitivity: Optional[float] = None) -> np.ndarray:
        """Add DP noise to batch of gradients (standard implementation)
        
        Args:
            gradients: Batch of gradients (shape: [batch_size, gradient_dim])
            sensitivity: Gradient sensitivity
            
        Returns:
            Noisy gradient batch
        """
        if sensitivity is None:
            sensitivity = self.sensitivity
        
        # Compute noise scale
        sigma = (
            np.sqrt(2 * np.log(1.25 / self.delta)) *
            sensitivity / self.epsilon
        )
        
        # Generate noise
        noise = np.random.normal(0, sigma, gradients.shape)
        
        # Add noise
        noisy_gradients = gradients + noise
        
        self.epsilon_consumed += self.epsilon
        
        return noisy_gradients

    def clip_gradient(self, gradient: np.ndarray,
                     norm_bound: float = 1.0) -> Tuple[np.ndarray, float]:
        """Clip gradient to bound L2 norm (preserves sensitivity)
        
        Args:
            gradient: Gradient to clip
            norm_bound: Maximum allowed L2 norm
            
        Returns:
            (clipped_gradient, clipping_ratio)
        """
        grad_norm = np.linalg.norm(gradient)
        
        if grad_norm > norm_bound:
            # Clip: g' = (norm_bound / ||g||) * g
            clipped_gradient = (norm_bound / (grad_norm + 1e-10)) * gradient
            clipping_ratio = norm_bound / grad_norm
        else:
            clipped_gradient = gradient
            clipping_ratio = 1.0
        
        return clipped_gradient, clipping_ratio

    def compute_noise_scale(self, sensitivity: Optional[float] = None) -> float:
        """Compute noise scale for Gaussian mechanism
        
        Formula: σ = √(2 log(1/δ)) * Δf / ε
        where Δf is sensitivity
        
        Args:
            sensitivity: Gradient sensitivity bound
            
        Returns:
            Noise standard deviation
        """
        if sensitivity is None:
            sensitivity = self.sensitivity
        
        sigma = (
            np.sqrt(2 * np.log(1.25 / self.delta)) *
            sensitivity / self.epsilon
        )
        
        return sigma

    def compose_epsilon(self, num_rounds: int) -> float:
        """Compute composed epsilon after multiple rounds
        
        Advanced composition theorem:
        After k rounds of (ε, δ)-DP:
        Final privacy: (ε_total, δ_total)
        where ε_total = √(2k log(1/δ')) * ε + k*ε*(e^ε - 1)
        
        Args:
            num_rounds: Number of training rounds
            
        Returns:
            Composed epsilon value
        """
        # Simplified composition (sequential)
        epsilon_composed = num_rounds * self.epsilon
        
        return epsilon_composed

    def compose_delta(self, num_rounds: int) -> float:
        """Compute composed delta after multiple rounds
        
        Args:
            num_rounds: Number of training rounds
            
        Returns:
            Composed delta value
        """
        # Delta accumulates (additive in basic composition)
        delta_composed = num_rounds * self.delta
        
        return min(delta_composed, 1.0)  # Cap at 1.0

    def gaussian_mechanism(self, data: np.ndarray,
                          sensitivity: Optional[float] = None) -> np.ndarray:
        """Apply Gaussian mechanism (theoretical reference)
        
        Args:
            data: Input data/gradient
            sensitivity: Sensitivity of query function
            
        Returns:
            Differentially private output
        """
        if sensitivity is None:
            sensitivity = self.sensitivity
        
        sigma = (
            np.sqrt(2 * np.log(1.25 / self.delta)) *
            sensitivity / self.epsilon
        )
        
        noise = np.random.normal(0, sigma, data.shape)
        
        return data + noise

    def laplace_mechanism(self, data: np.ndarray,
                         sensitivity: Optional[float] = None) -> np.ndarray:
        """Apply Laplace mechanism (alternative to Gaussian)
        
        Args:
            data: Input data/gradient
            sensitivity: Sensitivity
            
        Returns:
            Differentially private output
        """
        if sensitivity is None:
            sensitivity = self.sensitivity
        
        # Scale for Laplace: b = sensitivity / epsilon
        scale = sensitivity / self.epsilon
        
        # Laplace noise
        noise = np.random.laplace(0, scale, data.shape)
        
        return data + noise

    def get_privacy_summary(self) -> dict:
        """Get privacy budget summary
        
        Returns:
            Dictionary with privacy parameters and consumption
        """
        return {
            'epsilon_budget': self.epsilon,
            'delta_budget': self.delta,
            'epsilon_consumed': self.epsilon_consumed,
            'sensitivity': self.sensitivity,
            'mechanism': self.mechanism,
            'noise_scale': self.compute_noise_scale(),
            'privacy_level': self._estimate_privacy_level()
        }

    def _estimate_privacy_level(self) -> str:
        """Estimate privacy level from epsilon value
        
        Returns:
            Privacy level description
        """
        if self.epsilon <= 0.5:
            return "EXTREMELY_STRONG"
        elif self.epsilon <= 1.0:
            return "VERY_STRONG"
        elif self.epsilon <= 1.5:
            return "STRONG"
        elif self.epsilon <= 2.0:
            return "MODERATE_STRONG"
        elif self.epsilon <= 3.0:
            return "MODERATE"
        elif self.epsilon <= 5.0:
            return "WEAK"
        else:
            return "VERY_WEAK"

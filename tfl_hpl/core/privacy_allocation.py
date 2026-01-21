"""Personalized Privacy Budget Allocation

Allocates device-specific privacy budgets (ε_i, δ_i) based on:
1. Trustworthiness scores
2. Device hardware capabilities
3. Historical privacy consumption
4. Global privacy budget constraints

Formula: ε_i = ε_global × (trust_score_i / Σ trust_scores)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from loguru import logger


class PrivacyTier(Enum):
    """Privacy protection levels"""
    STRICT = 1.0      # ε ≤ 1.0 (strongest privacy)
    HIGH = 1.5        # 1.0 < ε ≤ 1.5
    STANDARD = 2.0    # 1.5 < ε ≤ 2.0
    RELAXED = 3.0     # 2.0 < ε ≤ 3.0
    MINIMAL = 5.0     # ε > 3.0 (weakest privacy)


@dataclass
class PrivacyBudget:
    """Per-device privacy budget"""
    device_id: int
    epsilon: float
    delta: float
    privacy_tier: PrivacyTier
    consumed_epsilon: float = 0.0
    remaining_epsilon: float = 0.0
    renewal_round: int = 0


class PrivacyAllocator:
    """Personalized Privacy Budget Allocation
    
    Implements Algorithm 1 (Lines 6-10):
    ε_i = ε_global × (trust_score_i / Σ trust_scores)
    δ_i = δ_global × (trust_score_i / K)
    
    Ensures:
    - Global privacy guarantee maintained through composition
    - High-trust devices receive stricter privacy (less noise)
    - Low-trust devices receive looser privacy (more noise to observe)
    - No single device can exceed global privacy budget
    """

    def __init__(self, epsilon_global: float = 2.0, 
                 delta_global: float = 1e-5,
                 num_devices: Optional[int] = None,
                 allow_renewal: bool = True):
        """Initialize privacy allocator
        
        Args:
            epsilon_global: Global privacy budget for entire federated system
            delta_global: Global delta parameter for (ε,δ)-differential privacy
            num_devices: Number of devices (optional, set during allocation)
            allow_renewal: Whether to allow privacy budget renewal between rounds
        """
        self.epsilon_global = epsilon_global
        self.delta_global = delta_global
        self.num_devices = num_devices
        self.allow_renewal = allow_renewal
        
        # Budget tracking
        self.device_budgets = {}  # device_id -> PrivacyBudget
        self.total_epsilon_consumed = 0.0
        self.allocation_history = []
        
        logger.info(f"PrivacyAllocator initialized: ε={epsilon_global}, δ={delta_global}")

    def allocate_epsilon(self, trust_scores: np.ndarray,
                        num_devices: Optional[int] = None) -> np.ndarray:
        """Allocate individual epsilon values based on trust scores (Algorithm 1, Line 9)
        
        Formula: ε_i = ε_global × (trust_score_i / Σ trust_scores)
        
        Args:
            trust_scores: Array of trust scores for devices [0, 1]
            num_devices: Override number of devices
            
        Returns:
            Array of device-specific epsilon values
        """
        if num_devices is not None:
            self.num_devices = num_devices
        
        if self.num_devices is None:
            self.num_devices = len(trust_scores)
        
        # Line 7: trust_sum ← Σ(i=1 to K) trust_scores[i]
        trust_sum = np.sum(trust_scores)
        
        if trust_sum < 1e-10:
            # All zero trust scores - uniform allocation
            logger.warning("All trust scores are zero, using uniform allocation")
            epsilon_budgets = np.ones(len(trust_scores)) * (self.epsilon_global / len(trust_scores))
        else:
            # Line 9: ε_i ← ε_global × (trust_scores[i] / trust_sum)
            epsilon_budgets = (
                self.epsilon_global * 
                (trust_scores / trust_sum)
            )
        
        # Ensure minimum privacy even for low-trust devices
        min_epsilon = self.epsilon_global / (10 * len(trust_scores))
        epsilon_budgets = np.maximum(epsilon_budgets, min_epsilon)
        
        # Renormalize to respect global budget
        epsilon_budgets = epsilon_budgets * (self.epsilon_global / np.sum(epsilon_budgets))
        
        return epsilon_budgets

    def allocate_delta(self, trust_scores: np.ndarray,
                      num_devices: Optional[int] = None) -> np.ndarray:
        """Allocate individual delta values based on trust scores (Algorithm 1, Line 10)
        
        Formula: δ_i = δ_global × (trust_score_i / K)
        
        Args:
            trust_scores: Array of trust scores [0, 1]
            num_devices: Number of devices
            
        Returns:
            Array of device-specific delta values
        """
        if num_devices is not None:
            self.num_devices = num_devices
        
        if self.num_devices is None:
            self.num_devices = len(trust_scores)
        
        # Line 10: δ_i ← δ_global × (trust_scores[i] / K)
        delta_budgets = (
            self.delta_global *
            (trust_scores / self.num_devices)
        )
        
        # Ensure non-zero delta for all devices
        delta_budgets = np.maximum(delta_budgets, self.delta_global / (1000 * self.num_devices))
        
        return delta_budgets

    def allocate_budgets(self, trust_scores: np.ndarray,
                        num_devices: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Allocate both epsilon and delta budgets
        
        Args:
            trust_scores: Device trust scores
            num_devices: Number of devices
            
        Returns:
            (epsilon_budgets, delta_budgets)
        """
        epsilon_budgets = self.allocate_epsilon(trust_scores, num_devices)
        delta_budgets = self.allocate_delta(trust_scores, num_devices)
        
        # Record allocation
        self.allocation_history.append({
            'trust_scores': trust_scores.copy(),
            'epsilon_budgets': epsilon_budgets.copy(),
            'delta_budgets': delta_budgets.copy(),
            'total_epsilon': np.sum(epsilon_budgets),
            'total_delta': np.sum(delta_budgets)
        })
        
        return epsilon_budgets, delta_budgets

    def get_privacy_tier(self, epsilon: float) -> PrivacyTier:
        """Get privacy tier for given epsilon value
        
        Args:
            epsilon: Privacy budget epsilon
            
        Returns:
            Privacy tier classification
        """
        if epsilon <= 1.0:
            return PrivacyTier.STRICT
        elif epsilon <= 1.5:
            return PrivacyTier.HIGH
        elif epsilon <= 2.0:
            return PrivacyTier.STANDARD
        elif epsilon <= 3.0:
            return PrivacyTier.RELAXED
        else:
            return PrivacyTier.MINIMAL

    def update_device_budget(self, device_id: int,
                            epsilon: float,
                            delta: float,
                            consumed_epsilon: float = 0.0) -> None:
        """Update privacy budget for a specific device
        
        Args:
            device_id: Device identifier
            epsilon: Allocated epsilon
            delta: Allocated delta
            consumed_epsilon: Amount already consumed
        """
        privacy_tier = self.get_privacy_tier(epsilon)
        
        self.device_budgets[device_id] = PrivacyBudget(
            device_id=device_id,
            epsilon=epsilon,
            delta=delta,
            privacy_tier=privacy_tier,
            consumed_epsilon=consumed_epsilon,
            remaining_epsilon=epsilon - consumed_epsilon
        )

    def consume_epsilon(self, device_id: int, amount: float) -> bool:
        """Track epsilon consumption for a device
        
        Args:
            device_id: Device identifier
            amount: Amount of epsilon to consume
            
        Returns:
            Whether consumption was successful (within budget)
        """
        if device_id not in self.device_budgets:
            logger.warning(f"Device {device_id} budget not initialized")
            return False
        
        budget = self.device_budgets[device_id]
        
        if budget.remaining_epsilon >= amount:
            budget.consumed_epsilon += amount
            budget.remaining_epsilon -= amount
            self.total_epsilon_consumed += amount
            return True
        else:
            logger.warning(
                f"Device {device_id}: Insufficient epsilon budget. "
                f"Requested {amount:.4f}, remaining {budget.remaining_epsilon:.4f}"
            )
            return False

    def get_device_budget(self, device_id: int) -> Optional[PrivacyBudget]:
        """Get privacy budget for a device
        
        Args:
            device_id: Device identifier
            
        Returns:
            Privacy budget or None if not allocated
        """
        return self.device_budgets.get(device_id)

    def renew_budgets(self, trust_scores: Optional[np.ndarray] = None) -> None:
        """Renew privacy budgets for new round
        
        Args:
            trust_scores: Updated trust scores (optional, uses previous if not provided)
        """
        if not self.allow_renewal:
            return
        
        if trust_scores is not None:
            epsilon_budgets, delta_budgets = self.allocate_budgets(trust_scores)
        else:
            # Use uniform renewal if no trust scores provided
            num_devices = len(self.device_budgets)
            epsilon_budgets = np.ones(num_devices) * (self.epsilon_global / num_devices)
            delta_budgets = np.ones(num_devices) * (self.delta_global / num_devices)
        
        for i, (device_id, budget) in enumerate(self.device_budgets.items()):
            budget.consumed_epsilon = 0.0
            budget.remaining_epsilon = epsilon_budgets[i]
            budget.renewal_round += 1

    def get_allocation_summary(self) -> Dict:
        """Get summary of privacy allocations
        
        Returns:
            Dictionary with allocation statistics
        """
        if not self.allocation_history:
            return {}
        
        latest = self.allocation_history[-1]
        
        return {
            'num_devices': len(latest['epsilon_budgets']),
            'total_epsilon': latest['total_epsilon'],
            'total_delta': latest['total_delta'],
            'avg_epsilon': np.mean(latest['epsilon_budgets']),
            'min_epsilon': np.min(latest['epsilon_budgets']),
            'max_epsilon': np.max(latest['epsilon_budgets']),
            'consumed_epsilon': self.total_epsilon_consumed,
            'global_epsilon': self.epsilon_global,
            'global_delta': self.delta_global
        }

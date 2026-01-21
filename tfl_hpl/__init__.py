"""
TFL-HPL: Trustworthy Federated Learning with Heterogeneous Privacy Levels

A Byzantine-resilient federated learning framework for critical infrastructure IoT systems.
"""

__version__ = "1.0.0"
__author__ = "Burra Deepak Yadav"
__email__ = "deepakyadavdeepu94@gmail.com"

from tfl_hpl.core.server import FLServerCoordinator
from tfl_hpl.core.client import FLClientDevice
from tfl_hpl.core.trust_scoring import TrustScorer
from tfl_hpl.core.byzantine_detection import ByzantineDetector
from tfl_hpl.core.privacy_allocation import PrivacyAllocator
from tfl_hpl.config.default_config import DefaultConfig

__all__ = [
    "FLServerCoordinator",
    "FLClientDevice",
    "TrustScorer",
    "ByzantineDetector",
    "PrivacyAllocator",
    "DefaultConfig",
]

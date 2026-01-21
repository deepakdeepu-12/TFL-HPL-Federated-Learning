"""Default Configuration for TFL-HPL

Centralized configuration management for all framework components.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class DeviceType(Enum):
    """Supported device types"""
    SCADA_CONTROLLER = "scada_controller"
    IOT_SENSOR = "iot_sensor"
    EDGE_GATEWAY = "edge_gateway"
    EDGE_SERVER = "edge_server"
    CLOUD_SERVER = "cloud_server"


class AggregationStrategy(Enum):
    """Aggregation strategies"""
    MEDIAN = "median"
    KRUM = "krum"
    MULTI_KRUM = "multi_krum"
    BULYAN = "bulyan"


@dataclass
class DefaultConfig:
    """Default TFL-HPL Configuration
    
    Contains all hyperparameters for federated learning setup.
    """
    
    # ====== SYSTEM CONFIGURATION ======
    num_devices: int = 50
    num_rounds: int = 500
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.01
    lr_decay: float = 1.0
    
    # ====== PRIVACY CONFIGURATION ======
    epsilon_global: float = 2.0  # Global privacy budget
    delta: float = 1e-5  # Failure probability
    gradient_clip: float = 1.0  # Gradient clipping norm
    
    # ====== BYZANTINE CONFIGURATION ======
    aggregation_strategy: AggregationStrategy = AggregationStrategy.MEDIAN
    byzantine_threshold: float = 2.0  # Divergence threshold for attack detection
    attack_threshold: float = 0.35  # Fraction threshold for privacy amplification
    privacy_amplification_factor: float = 0.8  # Epsilon multiplier during attack
    
    # ====== TRUST CONFIGURATION ======
    trust_consistency_weight: float = 0.4
    trust_anomaly_weight: float = 0.3
    trust_reliability_weight: float = 0.3
    trust_history_size: int = 100
    
    # ====== COMMUNICATION CONFIGURATION ======
    device_timeout: float = 30.0  # Device communication timeout (seconds)
    quantization_bits: int = 8  # Gradient quantization
    communication_rounds: int = 500
    
    # ====== HARDWARE CONFIGURATION ======
    device_types: List[DeviceType] = field(default_factory=lambda: [
        DeviceType.SCADA_CONTROLLER,
        DeviceType.IOT_SENSOR,
        DeviceType.EDGE_GATEWAY,
        DeviceType.EDGE_SERVER
    ])
    
    # Hardware specifications per device type
    hardware_specs: Dict = field(default_factory=lambda: {
        DeviceType.SCADA_CONTROLLER: {
            'memory_mb': 256,
            'cpu_ghz': 0.7,
            'streaming_mode': True
        },
        DeviceType.IOT_SENSOR: {
            'memory_mb': 512,
            'cpu_ghz': 1.2,
            'streaming_mode': True
        },
        DeviceType.EDGE_GATEWAY: {
            'memory_mb': 2048,
            'cpu_ghz': 2.0,
            'streaming_mode': False
        },
        DeviceType.EDGE_SERVER: {
            'memory_mb': 8192,
            'cpu_ghz': 2.8,
            'streaming_mode': False
        },
        DeviceType.CLOUD_SERVER: {
            'memory_mb': 65536,
            'cpu_ghz': 3.5,
            'streaming_mode': False
        }
    })
    
    # ====== DATASET CONFIGURATION ======
    dataset_name: str = "ieee_9_bus"
    train_test_split: float = 0.8
    data_augmentation: bool = False
    shuffle: bool = True
    
    # ====== LOGGING & MONITORING ======
    log_level: str = "INFO"
    log_interval: int = 10  # Log every N rounds
    save_model: bool = True
    save_metrics: bool = True
    output_dir: str = "./outputs"
    
    # ====== EXPERIMENT CONFIGURATION ======
    seed: int = 42
    device: str = "cuda"  # [cuda, cpu]
    num_workers: int = 4
    pin_memory: bool = True
    
    # ====== ATTACK SIMULATION ======
    byzantine_ratio: float = 0.0  # Fraction of Byzantine devices (0-0.33)
    attack_type: Optional[str] = None  # [label_flipping, gradient_inversion, model_poisoning]
    attack_intensity: float = 1.0  # Attack strength multiplier
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {
            'num_devices': self.num_devices,
            'num_rounds': self.num_rounds,
            'epsilon_global': self.epsilon_global,
            'delta': self.delta,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'local_epochs': self.local_epochs,
            'byzantine_threshold': self.byzantine_threshold,
            'gradient_clip': self.gradient_clip,
        }

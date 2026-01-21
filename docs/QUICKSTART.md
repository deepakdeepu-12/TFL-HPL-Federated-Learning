# TFL-HPL Quick Start Guide

## Installation

```bash
git clone https://github.com/deepakdeepu-12/TFL-HPL-Federated-Learning.git
cd TFL-HPL-Federated-Learning
pip install -r requirements.txt
pip install -e .
```

## Basic Usage

### 1. Simple Federated Learning

```python
from tfl_hpl.core.server import FLServerCoordinator, ServerConfig
from tfl_hpl.core.client import FLClientDevice, ClientConfig
from tfl_hpl.config.default_config import DefaultConfig
import torch
import torch.nn as nn

# 1. Define model
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 2. Initialize server
server_config = ServerConfig(
    num_devices=50,
    num_rounds=100,
    epsilon_global=2.0,
    delta=1e-5,
    learning_rate=0.01
)
server = FLServerCoordinator(server_config)

# 3. Initialize model
model = SimpleNN()
server.initialize_model(model)

# 4. Initialize clients
clients = []
for i in range(50):
    client_config = ClientConfig(
        device_id=i,
        learning_rate=0.01,
        local_epochs=5,
        batch_size=32
    )
    client = FLClientDevice(client_config)
    client.set_model(model)
    clients.append(client)

# 5. Run training
metrics = server.train(clients)

print(f"Training complete!")
print(f"Rounds: {len(metrics['trust_scores_history'])}")
print(f"Final model: {server.global_model}")
```

### 2. Run Experiments

```bash
# IEEE 9-bus smart grid
python experiments/ieee_9_bus.py --num_rounds 500 --num_devices 50

# With Byzantine attacks
python experiments/Byzantine_attacks.py --attack_type label_flipping --attack_ratio 0.1

# Ablation study
python experiments/ablation_study.py --all_components
```

## Configuration

Customize your experiment with `DefaultConfig`:

```python
from tfl_hpl.config.default_config import DefaultConfig, AggregationStrategy

config = DefaultConfig(
    num_devices=100,
    num_rounds=500,
    epsilon_global=1.8,  # Stricter privacy
    delta=1e-5,
    learning_rate=0.01,
    batch_size=32,
    local_epochs=5,
    aggregation_strategy=AggregationStrategy.MEDIAN,
    byzantine_threshold=2.0,
    attack_threshold=0.35,
)
```

## Key Components

### 1. Trust Scoring

```python
from tfl_hpl.core.trust_scoring import TrustScorer

trust_scorer = TrustScorer(
    consistency_weight=0.4,
    anomaly_weight=0.3,
    reliability_weight=0.3
)

trust_score = trust_scorer.compute_trust_score(
    device_id=0,
    device_gradient=client_grad,
    consensus_gradient=server_grad,
    quality=0.95
)
```

### 2. Byzantine Detection

```python
from tfl_hpl.core.byzantine_detection import ByzantineDetector

detector = ByzantineDetector(threshold=2.0)

is_attack, confidence = detector.detect_attack(
    device_id=0,
    device_gradient=client_grad,
    consensus_gradient=server_grad
)
```

### 3. Privacy Budget Allocation

```python
from tfl_hpl.core.privacy_allocation import PrivacyAllocator
import numpy as np

allocator = PrivacyAllocator(epsilon_global=2.0, delta=1e-5)

trust_scores = np.array([0.95, 0.70, 0.35])
epsilon_budgets, delta_budgets = allocator.allocate_budgets(trust_scores)

print(f"Device 0: ε={epsilon_budgets[0]:.4f}")
print(f"Device 1: ε={epsilon_budgets[1]:.4f}")
print(f"Device 2: ε={epsilon_budgets[2]:.4f}")
```

### 4. Differential Privacy

```python
from tfl_hpl.core.differential_privacy import DifferentialPrivacyEngine

dp = DifferentialPrivacyEngine(epsilon=1.8, delta=1e-5)

# Add noise to gradients
noisy_gradient = dp.add_noise_streaming(
    gradient=client_gradient,
    sensitivity=1.0
)

# Clip gradient
clipped_grad, ratio = dp.clip_gradient(client_gradient, norm_bound=1.0)
```

## Advanced Usage

### Heterogeneous Devices

```python
from tfl_hpl.config.default_config import DeviceType

# Create devices with different hardware
device_configs = [
    ClientConfig(device_id=i, device_type=DeviceType.SCADA_CONTROLLER.value)
    for i in range(10)
] + [
    ClientConfig(device_id=i, device_type=DeviceType.EDGE_GATEWAY.value)
    for i in range(10, 50)
]

# Server automatically adjusts privacy budgets based on device capabilities
```

### Attack Simulation

```python
from experiments.Byzantine_attacks import create_byzantine_client

# Create Byzantine client for testing
byzantine_client = create_byzantine_client(
    device_id=0,
    attack_type="label_flipping",
    attack_intensity=1.0
)

clients[0] = byzantine_client
```

## Monitoring Training

```python
metrics = server.train(clients)

print(f"Attack history: {metrics['attack_history']}")
print(f"Trust scores evolution: {metrics['trust_scores_history']}")
print(f"Total training time: {metrics['total_time']:.2f}s")
```

## Performance Tips

1. **GPU Acceleration**: Set `device='cuda'` in config
2. **Quantization**: Use `quantization_bits=8` to reduce communication
3. **Streaming Mode**: Enable for SCADA devices to save memory
4. **Batch Size**: Increase for faster convergence (trades off variance)
5. **Local Epochs**: Increase for better model quality (trades off communication)

## Troubleshooting

### Out of Memory
- Reduce `batch_size`
- Enable `streaming_mode=True` for SCADA devices
- Reduce `quantization_bits`

### Slow Convergence
- Increase `learning_rate` (carefully)
- Increase `local_epochs`
- Check for Byzantine attacks (unusually high `attack_history`)

### Privacy Too Weak
- Decrease `epsilon_global` (e.g., 1.8 → 1.0)
- This will add more DP noise and slow convergence

## Next Steps

- Read [API_REFERENCE.md](API_REFERENCE.md) for detailed API documentation
- See [DEPLOYMENT.md](DEPLOYMENT.md) for production deployment
- Check [experiments/](../experiments/) directory for more examples

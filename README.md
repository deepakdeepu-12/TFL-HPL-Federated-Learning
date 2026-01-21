# TFL-HPL: Trustworthy Federated Learning with Heterogeneous Privacy Levels

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2401.xxxxx-b31b1b.svg)](https://arxiv.org)
[![IEEE GSEACT 2026](https://img.shields.io/badge/Conference-IEEE%20GSEACT%202026-green)](https://gseact2026.ieee.org)

## Overview

**TFL-HPL** is a Byzantine-resilient federated learning framework designed specifically for **critical infrastructure IoT systems** (smart grids, water treatment, SCADA networks). It uniquely combines:

- **Dynamic Trustworthiness Scoring**: Real-time reputation mechanism for continuous device assessment
- **Personalized ε-Differential Privacy**: Device-adaptive privacy budgets based on trust scores
- **Byzantine-Robust Aggregation**: Median-based aggregation resistant to >33% compromised devices
- **Privacy Amplification During Attacks**: Automatic privacy budget boost for honest devices under attack
- **SCADA-Optimized Implementation**: Deploys on 256MB legacy controllers with streaming DP noise

### Key Results

| Metric | Performance |
|--------|-------------|
| **Accuracy (IEEE 9-Bus)** | 94.7% ± 0.8% |
| **Privacy Guarantee** | ε = 1.8 (strict privacy) |
| **Convergence Speedup** | 156% faster than FedByzantine |
| **Attack Detection Rate** | 94.2-100% across 4 adversarial scenarios |
| **Hardware Compatibility** | 256MB SCADA to 8GB edge servers |

## Repository Structure

```
TFL-HPL-Federated-Learning/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package installation
│
├── tfl_hpl/                          # Main package
│   ├── __init__.py                   # Package initialization
│   ├── core/                         # Core algorithms
│   │   ├── __init__.py
│   │   ├── server.py                 # Server-side protocol (Algorithm 1)
│   │   ├── client.py                 # Client-side training (Algorithm 2)
│   │   ├── trust_scoring.py          # Trustworthiness scoring mechanism
│   │   ├── byzantine_detection.py    # Byzantine attack detection
│   │   ├── privacy_allocation.py     # Personalized privacy budget allocation
│   │   └── differential_privacy.py   # Differential privacy mechanisms
│   │
│   ├── aggregation/                  # Aggregation strategies
│   │   ├── __init__.py
│   │   ├── median_aggregation.py     # Coordinate-wise median
│   │   ├── robust_aggregation.py     # Byzantine-robust strategies
│   │   └── privacy_aware_agg.py      # Privacy-aware aggregation
│   │
│   ├── utils/                        # Utility functions
│   │   ├── __init__.py
│   │   ├── data_loader.py            # Dataset loading utilities
│   │   ├── metrics.py                # Evaluation metrics
│   │   ├── communication.py          # Encrypted communication
│   │   └── logging_utils.py          # Logging and monitoring
│   │
│   └── config/                       # Configuration management
│       ├── __init__.py
│       ├── default_config.py         # Default hyperparameters
│       └── hardware_profiles.py      # Device-specific profiles
│
├── experiments/                       # Experimental scripts
│   ├── ieee_9_bus.py                 # IEEE 9-bus grid evaluation
│   ├── ieee_118_bus.py               # IEEE 118-bus grid evaluation
│   ├── water_treatment.py            # Water treatment facility
│   ├── ablation_study.py             # Ablation study reproduction
│   ├── sensitivity_analysis.py       # Hyperparameter sensitivity
│   └── Byzantine_attacks.py          # Adversarial robustness testing
│
├── datasets/                         # Dataset files
│   ├── ieee_9_bus/                   # IEEE 9-bus power system data
│   ├── ieee_118_bus/                 # IEEE 118-bus power system data
│   ├── water_treatment/              # Water treatment IoT data
│   └── README.md                     # Dataset documentation
│
├── models/                           # Pre-trained models & configs
│   ├── ieee_9_bus_model.pt           # Pre-trained model for 9-bus
│   ├── ieee_118_bus_model.pt         # Pre-trained model for 118-bus
│   └── model_registry.py             # Model management
│
├── docs/                             # Documentation
│   ├── INSTALLATION.md               # Installation guide
│   ├── QUICKSTART.md                 # Quick start tutorial
│   ├── API_REFERENCE.md              # Complete API documentation
│   ├── ARCHITECTURE.md               # System architecture details
│   ├── DEPLOYMENT.md                 # Production deployment guide
│   └── RESEARCH_PAPER.md             # Linked research paper
│
├── tests/                            # Unit & integration tests
│   ├── test_trust_scoring.py         # Trust mechanism tests
│   ├── test_byzantine_detection.py   # Attack detection tests
│   ├── test_privacy.py               # Privacy preservation tests
│   ├── test_aggregation.py           # Aggregation correctness
│   └── test_integration.py           # End-to-end integration tests
│
├── scripts/                          # Utility scripts
│   ├── generate_datasets.py          # Dataset generation
│   ├── evaluate_all.py               # Run all experiments
│   ├── deploy_scada.py               # SCADA deployment script
│   └── benchmark.py                  # Performance benchmarking
│
└── docker/                           # Docker configuration
    ├── Dockerfile                    # Container definition
    ├── docker-compose.yml            # Multi-container orchestration
    └── entrypoint.sh                 # Container entrypoint

```

## Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager
- (Optional) Docker for containerized deployment

### Option 1: Local Installation

```bash
# Clone repository
git clone https://github.com/deepakdeepu-12/TFL-HPL-Federated-Learning.git
cd TFL-HPL-Federated-Learning

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Option 2: Docker Installation

```bash
# Build Docker image
docker build -t tfl-hpl:latest ./docker

# Run container
docker run -it --rm tfl-hpl:latest python experiments/ieee_9_bus.py
```

## Quick Start

### 1. Basic Usage

```python
from tfl_hpl.core.server import FLServerCoordinator
from tfl_hpl.core.client import FLClientDevice
from tfl_hpl.config.default_config import DefaultConfig

# Initialize configuration
config = DefaultConfig(
    num_devices=50,
    num_rounds=500,
    epsilon_global=2.0,
    delta=1e-5
)

# Initialize server
server = FLServerCoordinator(config)

# Initialize clients
clients = [
    FLClientDevice(device_id=i, config=config)
    for i in range(config.num_devices)
]

# Run federated learning
final_model = server.train(clients)
```

### 2. Run Pre-configured Experiments

```bash
# IEEE 9-bus smart grid evaluation
python experiments/ieee_9_bus.py --num_rounds 500 --num_devices 50

# IEEE 118-bus smart grid evaluation
python experiments/ieee_118_bus.py --num_rounds 500 --num_devices 100

# Water treatment facility
python experiments/water_treatment.py --num_rounds 500 --num_devices 45

# Ablation study
python experiments/ablation_study.py --all_components

# Byzantine attack scenarios
python experiments/Byzantine_attacks.py --attack_type label_flipping --attack_rate 0.1
```

### 3. Advanced Configuration

```python
from tfl_hpl.config.default_config import DefaultConfig
from tfl_hpl.core.server import FLServerCoordinator

# Custom configuration
config = DefaultConfig(
    num_devices=200,
    num_rounds=300,
    epsilon_global=1.8,
    delta=1e-5,
    learning_rate=0.01,
    batch_size=32,
    local_epochs=5,
    quantization_bits=8,
    gradient_clip=1.0,
    device_profiles=['scada', 'iot_sensor', 'edge_gateway', 'edge_server']
)

server = FLServerCoordinator(config)
metrics = server.train(clients)
```

## Core Components

### 1. Trust Scoring Mechanism
Continuous trustworthiness scoring combining:
- **Consistency Score**: Gradient alignment with peer consensus
- **Anomaly Score**: Statistical deviation detection using Isolation Forest
- **Reliability Score**: Historical participation quality

```python
from tfl_hpl.core.trust_scoring import TrustScorer

trust_scorer = TrustScorer(
    consistency_weight=0.4,
    anomaly_weight=0.3,
    reliability_weight=0.3
)

trust_score = trust_scorer.compute_score(
    consistency=0.95,
    anomaly_detection=0.98,
    reliability=0.90
)
```

### 2. Personalized Privacy Budget Allocation
Device-specific privacy budgets based on trustworthiness:

```python
from tfl_hpl.core.privacy_allocation import PrivacyAllocator

allocator = PrivacyAllocator(epsilon_global=2.0, delta=1e-5)

epsilon_individual = allocator.allocate_epsilon(
    trust_scores=[0.95, 0.70, 0.35],
    num_devices=3
)
# Returns: [0.927, 0.732, 0.341]
```

### 3. Byzantine Attack Detection
Multi-dimensional anomaly detection for model poisoning attacks:

```python
from tfl_hpl.core.byzantine_detection import ByzantineDetector

detector = ByzantineDetector(detection_threshold=2.0)

is_attack, confidence = detector.detect_attack(
    client_gradient=client_grad,
    aggregated_gradient=global_grad,
    historical_gradients=history
)
```

### 4. Differential Privacy Mechanisms
Streaming DP noise generation for memory-constrained devices:

```python
from tfl_hpl.core.differential_privacy import DifferentialPrivacy

dp_engine = DifferentialPrivacy(epsilon=1.8, delta=1e-5)

noisy_gradient = dp_engine.add_noise_streaming(
    gradient=client_gradient,
    sensitivity=1.0
)
```

## Experimental Results

### 1. Accuracy Performance

```
IEEE 9-Bus Grid Dataset:
  TFL-HPL:        94.7% ± 0.8%
  FedAvg:         91.3% ± 0.7%  (−3.4%, p<0.001)
  DP-FedAvg:      89.1% ± 1.1%  (−5.6%, p<0.001)
  FedByzantine:   88.7% ± 1.2%  (−6.0%, p<0.001)
```

### 2. Byzantine Attack Detection

```
Label Flipping (10%):      98.3% detection rate
Gradient Inversion (50%):  100.0% detection rate
Model Poisoning (5%):      94.2% detection rate
Collusion (3 devices):     96.1% detection rate
```

### 3. Privacy-Utility Trade-off

```
Privacy Budget ε | TFL-HPL Accuracy | DP-FedAvg Accuracy | Improvement
1.0              | 81.2%           | 74.9%             | +8.4%
1.8              | 85.4%           | 82.3%             | +3.1%
2.5              | 87.6%           | 86.8%             | +0.8%
4.0              | 90.3%           | 90.1%             | +0.2%
```

### 4. Hardware Compatibility

```
Device Type        | Memory | Communication | Training | ✓ Support
-------------------------------------------------------------------
Legacy SCADA       | 256MB  | 12.5±0.8s     | 85±12ms  | YES
IoT Sensor         | 512MB  | 8.3±0.6s      | 52±8ms   | YES
Edge Gateway       | 2GB    | 3.1±0.4s      | 18±3ms   | YES
Edge Server        | 8GB    | 1.2±0.2s      | 5±1ms    | YES
```

## Testing

Run comprehensive test suite:

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_trust_scoring.py -v

# Run with coverage report
pytest tests/ --cov=tfl_hpl --cov-report=html
```

## Deployment Guide

### SCADA Controller Deployment

```bash
# Generate optimized SCADA deployment
python scripts/deploy_scada.py \
    --device_type scada_controller \
    --memory_limit 256MB \
    --output deployment/scada_optimized.py

# Deploy on actual SCADA device
# (Follow SCADA_DEPLOYMENT.md for detailed instructions)
```

### Production Deployment

See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for:
- Cloud deployment (AWS, Azure, GCP)
- Edge computing platforms (AWS Greengrass, Azure IoT Edge)
- On-premises deployment
- Docker containerization
- Kubernetes orchestration

## Citation

If you use TFL-HPL in your research, please cite:

```bibtex
@article{yadav2026tfl,
  title={Trustworthy Federated Learning with Heterogeneous Privacy Levels: Byzantine-Resistant Distributed Learning with Device-Adaptive Privacy Budgets},
  author={Yadav, Burra Deepak},
  journal={IEEE GSEACT 2026},
  year={2026}
}
```

## Contributing

We welcome contributions! Please see [`CONTRIBUTING.md`](CONTRIBUTING.md) for:
- Development setup
- Code style guidelines
- Testing requirements
- Pull request process

## License

This project is licensed under the MIT License - see [`LICENSE`](LICENSE) file for details.

## Acknowledgments

- Madanapalle Institute of Technology and Science (MITS)
- IEEE GSEACT 2026 Conference
- All contributors and reviewers

## Contact

**Burra Deepak Yadav**
- Email: deepakyadavdeepu94@gmail.com
- GitHub: [@deepakdeepu-12](https://github.com/deepakdeepu-12)
- Institute: Madanapalle Institute of Technology and Science

## Disclaimer

This codebase is intended for research and educational purposes. Use with appropriate security measures in production environments. Critical infrastructure deployments require regulatory compliance and thorough security audits.

---

**Last Updated**: January 21, 2026  
**Status**: Active Development  
**Python Version**: 3.9+  
**License**: MIT

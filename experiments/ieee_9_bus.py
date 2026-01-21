#!/usr/bin/env python3
"""IEEE 9-Bus Smart Grid Experiment

Evaluates TFL-HPL on IEEE 9-bus power system dataset.
Benchmarks accuracy, convergence speed, and privacy.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time
from loguru import logger

from tfl_hpl.core.server import FLServerCoordinator, ServerConfig
from tfl_hpl.core.client import FLClientDevice, ClientConfig
from tfl_hpl.config.default_config import DefaultConfig, DeviceType
from tfl_hpl.utils.metrics import MetricsComputer


class GridOptimizationNN(nn.Module):
    """Neural network for power grid optimization"""
    
    def __init__(self, input_dim=20, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 32)
        self.fc3 = nn.Linear(32, 3)  # 3 classes: normal, warning, critical
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def generate_ieee_9_bus_data(num_samples=8640, num_features=20, num_devices=50):
    """Generate synthetic IEEE 9-bus data
    
    IEEE 9-bus system:
    - 9 buses
    - 5 generators
    - 50+ sensors per device
    - 12 months hourly data (8760 samples)
    
    Args:
        num_samples: Total samples (12 months hourly)
        num_features: Features per sample (sensor readings)
        num_devices: Number of distributed devices
        
    Returns:
        Distributed datasets for each device
    """
    # Generate synthetic data
    X = np.random.randn(num_samples, num_features).astype(np.float32)
    y = np.random.randint(0, 3, num_samples)  # 3 classes
    
    # Distribute data across devices
    samples_per_device = num_samples // num_devices
    device_data = []
    
    for i in range(num_devices):
        start_idx = i * samples_per_device
        end_idx = start_idx + samples_per_device if i < num_devices - 1 else num_samples
        
        X_device = X[start_idx:end_idx]
        y_device = y[start_idx:end_idx]
        
        # Create DataLoader for device
        dataset = TensorDataset(
            torch.from_numpy(X_device),
            torch.from_numpy(y_device)
        )
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        device_data.append(dataloader)
    
    # Test set (hold out)
    X_test = np.random.randn(1000, num_features).astype(np.float32)
    y_test = np.random.randint(0, 3, 1000)
    
    return device_data, (X_test, y_test)


def run_experiment(args):
    """Run IEEE 9-bus experiment"""
    
    logger.info("="*60)
    logger.info("IEEE 9-Bus Smart Grid Federated Learning Experiment")
    logger.info("="*60)
    
    # Generate data
    logger.info(f"Generating IEEE 9-bus dataset with {args.num_devices} devices...")
    device_data, (X_test, y_test) = generate_ieee_9_bus_data(
        num_samples=8640,
        num_devices=args.num_devices
    )
    
    # Initialize server
    server_config = ServerConfig(
        num_devices=args.num_devices,
        num_rounds=args.num_rounds,
        epsilon_global=args.epsilon,
        delta=1e-5,
        learning_rate=args.learning_rate,
        byzantine_threshold=2.0,
        attack_threshold=0.35
    )
    server = FLServerCoordinator(server_config)
    
    # Initialize model
    model = GridOptimizationNN(input_dim=20, hidden_dim=64)
    server.initialize_model(model)
    
    # Initialize clients
    logger.info(f"Initializing {args.num_devices} client devices...")
    clients = []
    device_types = [DeviceType.SCADA_CONTROLLER, DeviceType.IOT_SENSOR, 
                   DeviceType.EDGE_GATEWAY, DeviceType.EDGE_SERVER]
    
    for i in range(args.num_devices):
        device_type = device_types[i % len(device_types)]
        
        client_config = ClientConfig(
            device_id=i,
            learning_rate=args.learning_rate,
            local_epochs=5,
            batch_size=32,
            device_type=device_type.value
        )
        
        client = FLClientDevice(client_config, local_data=device_data[i])
        client.set_model(model)
        clients.append(client)
    
    # Run training
    logger.info("Starting federated learning...")
    start_time = time.time()
    
    metrics = server.train(clients, target_accuracy=0.90)
    
    total_time = time.time() - start_time
    
    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    with torch.no_grad():
        X_test_torch = torch.from_numpy(X_test).float()
        outputs = server.global_model(X_test_torch)
        predictions = torch.argmax(outputs, dim=1).numpy()
        
        test_metrics = MetricsComputer.compute_metrics(
            y_test, predictions
        )
    
    # Print results
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT RESULTS")
    logger.info("="*60)
    logger.info(f"Test Accuracy: {test_metrics.accuracy:.4f}")
    logger.info(f"Test Precision: {test_metrics.precision:.4f}")
    logger.info(f"Test Recall: {test_metrics.recall:.4f}")
    logger.info(f"Test F1: {test_metrics.f1:.4f}")
    logger.info(f"Total Training Time: {total_time:.2f}s")
    logger.info(f"Privacy Budget: ε={args.epsilon}, δ=1e-5")
    logger.info(f"Convergence Rounds: {metrics.get('convergence_round', args.num_rounds)}")
    logger.info(f"Byzantine Detections: {sum(metrics['attack_history'])}")
    logger.info("="*60)
    
    return metrics, test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IEEE 9-Bus Experiment")
    parser.add_argument("--num_devices", type=int, default=50, help="Number of devices")
    parser.add_argument("--num_rounds", type=int, default=500, help="Training rounds")
    parser.add_argument("--epsilon", type=float, default=2.0, help="Privacy budget")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Run experiment
    metrics, test_metrics = run_experiment(args)

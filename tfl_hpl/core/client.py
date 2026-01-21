"""TFL-HPL Client-Side Training (Algorithm 2)

Implements local training on individual devices with:
- Local model training with gradient clipping
- Streaming differential privacy noise generation
- Memory-efficient implementation for SCADA devices
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import time
from loguru import logger


@dataclass
class ClientConfig:
    """Client training configuration"""
    device_id: int
    learning_rate: float = 0.01
    local_epochs: int = 5
    batch_size: int = 32
    gradient_clip: float = 1.0
    device_type: str = "edge_gateway"  # [scada, iot_sensor, edge_gateway, server]


class FLClientDevice:
    """Client-Side Local Training (Algorithm 2)
    
    Implements local training on heterogeneous devices with:
    - Gradient clipping for DP
    - Streaming DP noise (memory-efficient)
    - Quality metrics computation
    """

    def __init__(self, config: ClientConfig, local_data: Optional[DataLoader] = None):
        """Initialize FL client
        
        Args:
            config: Client configuration
            local_data: Local dataset as DataLoader
        """
        self.config = config
        self.device_id = config.device_id
        self.local_data = local_data
        self.local_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.previous_loss = None
        
        logger.info(f"Client {self.device_id} initialized (type={config.device_type})")

    def set_model(self, model: nn.Module) -> None:
        """Set local model for training
        
        Args:
            model: Neural network model to train locally
        """
        self.local_model = model.to(self.device)

    def local_train(self, broadcast_pkg: Dict, 
                   epsilon_i: float, 
                   delta_i: float,
                   timeout: float = 30.0) -> Dict:
        """Perform local training with differential privacy (Algorithm 2)
        
        Args:
            broadcast_pkg: Global model and hyperparameters from server
            epsilon_i: Device-specific privacy budget
            delta_i: Device-specific delta parameter
            timeout: Maximum training time in seconds
            
        Returns:
            Update package with model, gradients, and quality metrics
            
        Implementation:
            - Lines 1-9: Initialize with global model
            - Lines 10-15: Compute gradients with clipping
            - Lines 17-20: Add DP noise using streaming mechanism
            - Lines 22-23: Update local model
            - Lines 26-28: Compute quality metrics
        """
        train_start = time.time()
        
        # Line 1: w_i ← w (initialize with global model)
        global_model = broadcast_pkg['model']
        learning_rate = broadcast_pkg['learning_rate']
        self._load_model_dict(global_model)
        
        if self.local_model is None:
            raise ValueError(f"Client {self.device_id}: Model not set")
        
        optimizer = torch.optim.SGD(
            self.local_model.parameters(),
            lr=learning_rate
        )
        criterion = nn.CrossEntropyLoss()
        
        local_updates = []
        total_loss = 0.0
        total_samples = 0
        
        # Line 2: for epoch = 1 to local_epochs do
        for epoch in range(self.config.local_epochs):
            if time.time() - train_start > timeout:
                logger.warning(f"Client {self.device_id}: Training timeout")
                break
            
            epoch_loss = 0.0
            epoch_samples = 0
            
            # Line 3: for each batch in local data
            if self.local_data is not None:
                for batch_idx, (X, y) in enumerate(self.local_data):
                    X, y = X.to(self.device), y.to(self.device)
                    
                    # Line 8: Compute loss
                    optimizer.zero_grad()
                    outputs = self.local_model(X)
                    loss = criterion(outputs, y)
                    
                    # Line 9: Compute gradients
                    loss.backward()
                    
                    # Line 10-15: CLIP GRADIENTS (for differential privacy)
                    total_norm = 0.0
                    for p in self.local_model.parameters():
                        if p.grad is not None:
                            total_norm += p.grad.data.norm(2).item() ** 2
                    total_norm = np.sqrt(total_norm)
                    
                    # Line 12: Compute clipping threshold
                    clip_coeff = self.config.gradient_clip / (total_norm + 1e-6)
                    if clip_coeff < 1.0:
                        for p in self.local_model.parameters():
                            if p.grad is not None:
                                # Line 14: Clip to norm C=1.0
                                p.grad.data.mul_(clip_coeff)
                    
                    # Line 17-20: ADD DP NOISE (Streaming mechanism - O(1) memory)
                    # Compute noise scale for streaming DP
                    sigma_i = (
                        self.config.gradient_clip * 
                        np.sqrt(2 * np.log(1.0 / delta_i)) / epsilon_i
                    )
                    
                    # Add noise directly to gradients (streaming, no need to store full noise vector)
                    for p in self.local_model.parameters():
                        if p.grad is not None:
                            # Line 19: noise_j ← Gaussian(0, σ_i) for each parameter
                            noise = torch.randn_like(p.grad) * sigma_i
                            p.grad.add_(noise)
                    
                    # Line 23: Update local model
                    optimizer.step()
                    
                    epoch_loss += loss.item() * X.size(0)
                    epoch_samples += X.size(0)
            
            total_loss += epoch_loss
            total_samples += epoch_samples
        
        # Line 26-28: Compute quality metrics
        quality = self._compute_quality_metrics(total_loss, total_samples)
        
        # Extract model updates
        model_dict = self._get_model_dict()
        
        update_package = {
            'device_id': self.device_id,
            'model': model_dict,
            'loss': total_loss / max(total_samples, 1),
            'quality': quality,
            'training_time': time.time() - train_start
        }
        
        logger.debug(f"Client {self.device_id}: Local training complete "
                    f"(loss={update_package['loss']:.4f}, quality={quality:.4f})")
        
        return update_package

    def _load_model_dict(self, model_dict: Dict) -> None:
        """Load model state from dictionary
        
        Args:
            model_dict: Dictionary of model parameters
        """
        if self.local_model is None:
            raise ValueError("Local model not initialized")
        
        for name, param in self.local_model.named_parameters():
            if name in model_dict:
                if isinstance(model_dict[name], torch.Tensor):
                    param.data = model_dict[name].clone().to(self.device)
                else:
                    param.data = torch.from_numpy(model_dict[name]).float().to(self.device)

    def _get_model_dict(self) -> Dict:
        """Extract model as dictionary
        
        Returns:
            Dictionary of model parameters
        """
        if self.local_model is None:
            raise ValueError("Local model not initialized")
        
        return {
            name: param.data.clone().cpu()
            for name, param in self.local_model.named_parameters()
        }

    def _compute_quality_metrics(self, loss: float, num_samples: int) -> float:
        """Compute update quality metric
        
        Args:
            loss: Training loss
            num_samples: Number of training samples
            
        Returns:
            Quality score (0-1, higher is better)
        """
        # Normalize loss to quality metric
        # Lower loss = higher quality
        avg_loss = loss / max(num_samples, 1)
        
        # Quality decreases exponentially with loss
        quality = np.exp(-avg_loss)
        
        # Store for future comparisons
        self.previous_loss = avg_loss
        
        return min(1.0, max(0.0, quality))

    def set_local_data(self, data_loader: DataLoader) -> None:
        """Set local training data
        
        Args:
            data_loader: PyTorch DataLoader for local data
        """
        self.local_data = data_loader

    def get_device_info(self) -> Dict:
        """Get device hardware information
        
        Returns:
            Device information including memory, CPU type, etc.
        """
        import psutil
        import GPUtil
        
        device_info = {
            'device_id': self.device_id,
            'device_type': self.config.device_type,
            'cpu_cores': psutil.cpu_count(),
            'ram_gb': psutil.virtual_memory().total / (1024**3),
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
        }
        
        # GPU info if available
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                device_info['gpu_available'] = True
                device_info['gpu_memory_gb'] = gpus[0].memoryTotal / 1024
            else:
                device_info['gpu_available'] = False
        except:
            device_info['gpu_available'] = False
        
        return device_info

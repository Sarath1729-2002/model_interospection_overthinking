"""
Loss function utilities for training.
"""

import torch
import torch.nn as nn


def create_loss_function(loss_fn_name: str) -> nn.Module:
    """
    Create a loss function based on the provided name.
    
    Args:
        loss_fn_name: One of 'mse', 'mae', 'log_mse', 'log_mae', 'log_mape', 'mape'
        
    Returns:
        A PyTorch loss function module
    """
    loss_fn_name = loss_fn_name.lower().strip()
    
    if loss_fn_name == 'mse':
        return nn.MSELoss()
    elif loss_fn_name == 'mae':
        return nn.L1Loss()
    elif loss_fn_name == 'mape':
        return MAPELoss()
    elif loss_fn_name == 'log_mse':
        return LogMSELoss()
    elif loss_fn_name == 'log_mae':
        return LogMAELoss()
    elif loss_fn_name == 'log_mape':
        return LogMAPELoss()
    else:
        raise ValueError(
            f"Unknown loss function: {loss_fn_name}. "
            f"Supported options: 'mse', 'mae', 'log_mse', 'log_mae', 'log_mape', 'mape'."
        )


class LogMSELoss(nn.Module):
    """Log of Mean Squared Error."""
    
    def __init__(self, epsilon: float = 1e-6):
        """
        Args:
            epsilon: Small value to add to MSE before taking log to avoid log(0)
        """
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute log MSE loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Loss value (log of MSE)
        """
        mse = torch.mean((predictions - targets) ** 2)
        return torch.log(mse + self.epsilon)


class LogMAELoss(nn.Module):
    """Log of Mean Absolute Error."""
    
    def __init__(self, epsilon: float = 1e-6):
        """
        Args:
            epsilon: Small value to add to MAE before taking log to avoid log(0)
        """
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute log MAE loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Loss value (log of MAE)
        """
        mae = torch.mean(torch.abs(predictions - targets))
        return torch.log(mae + self.epsilon)


class LogMAPELoss(nn.Module):
    """Log of Mean Absolute Percentage Error."""
    
    def __init__(self, epsilon: float = 1e-6):
        """
        Args:
            epsilon: Small value to add to MAE before taking log to avoid log(0)
        """
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute log MAE loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Loss value (log of MAPE)
        """
        mape = torch.mean(torch.abs(predictions - targets)/(targets + self.epsilon))
        return torch.log(mape + self.epsilon)
    
class MAPELoss(nn.Module):
    """Mean Absolute Percentage Error."""
    
    def __init__(self, epsilon: float = 1e-6):
        """
        Args:
            epsilon: Small value to add to targets to avoid division by zero
        """
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute MAPE loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Loss value (MAPE)
        """
        mape = torch.mean(torch.abs(predictions - targets)/(targets + self.epsilon))
        return mape
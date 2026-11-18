"""CPU regression model for the covariance experts' realised log variances."""
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn


HIDDEN_WIDTHS = (16, 8)


@dataclass
class NeuralRouter:
    network: nn.Sequential
    scaler: StandardScaler
    best_epoch: int
    epochs_trained: int
    validation_mse: float

    @torch.inference_mode()
    def predict(self, features: np.ndarray) -> np.ndarray:
        scaled = torch.as_tensor(self.scaler.transform(features), dtype=torch.float32)
        return self.network(scaled).numpy()


def fit_neural_router(train_features: np.ndarray, train_targets: np.ndarray,
                      validation_features: np.ndarray, validation_targets: np.ndarray,
                      *, seed: int, learning_rate: float, weight_decay: float,
                      max_epochs: int, patience: int) -> NeuralRouter:
    """Fit with training MSE and restore the epoch with the lowest validation MSE."""
    if max_epochs < 1 or patience < 1:
        raise ValueError("max_epochs and patience must be positive.")

    scaler = StandardScaler().fit(train_features)
    x_train = torch.as_tensor(scaler.transform(train_features), dtype=torch.float32)
    x_val = torch.as_tensor(scaler.transform(validation_features), dtype=torch.float32)
    y_train = torch.as_tensor(train_targets, dtype=torch.float32)
    y_val = torch.as_tensor(validation_targets, dtype=torch.float32)

    first, second = HIDDEN_WIDTHS
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        network = nn.Sequential(nn.Linear(x_train.shape[1], first), nn.ReLU(),
                                nn.Linear(first, second), nn.ReLU(),
                                nn.Linear(second, y_train.shape[1]))
    with torch.no_grad():
        network[-1].bias.copy_(y_train.mean(dim=0))

    optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    best_state = deepcopy(network.state_dict())
    best_epoch = 0
    best_mse = float("inf")
    for epoch in range(1, max_epochs + 1):
        network.train()
        optimizer.zero_grad()
        loss_fn(network(x_train), y_train).backward()
        optimizer.step()

        network.eval()
        with torch.no_grad():
            validation_mse = loss_fn(network(x_val), y_val).item()
        if not np.isfinite(validation_mse):
            raise RuntimeError("Neural router produced a non-finite validation loss.")
        if validation_mse < best_mse:
            best_state = deepcopy(network.state_dict())
            best_epoch = epoch
            best_mse = validation_mse
        if epoch - best_epoch >= patience:
            break

    network.load_state_dict(best_state)
    return NeuralRouter(network, scaler, best_epoch, epoch, best_mse)

"""Risk and turnover training through the differentiable GMV allocation."""
from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

from differentiable_gmv import DEFAULT_JITTER, DifferentiableGMV, covariance_mixture
from neural_router import HIDDEN_WIDTHS


TURNOVER_SMOOTHING = 1e-4  # Weight fraction.
RISK_SCALE = 252 * 1e4  # Annualised percentage-point squared.


@dataclass(frozen=True)
class PortfolioConfig:
    seed: int = 42
    temperature: float = .75
    jitter: float = DEFAULT_JITTER
    turnover_lambda: float = 0.
    learning_rate: float = .003
    weight_decay: float = .01
    max_epochs: int = 60
    patience: int = 12


class PortfolioAwareMLP(nn.Module):
    def __init__(self, features: int, assets: int, experts: int, config: PortfolioConfig):
        super().__init__()
        self.temperature = config.temperature
        first, second = HIDDEN_WIDTHS
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.seed)
            self.network = nn.Sequential(nn.Linear(features, first), nn.ReLU(),
                                         nn.Linear(first, second), nn.ReLU(), nn.Linear(second, experts)).double()
        self.gmv = DifferentiableGMV(assets, config.jitter)

    def forward(self, features: torch.Tensor, covariances: torch.Tensor):
        covariance, probabilities = covariance_mixture(self.network(features), covariances, self.temperature)
        return self.gmv(covariance), probabilities


def buy_and_hold(returns: torch.Tensor, weights: torch.Tensor):
    """Compute daily portfolio returns and final drifted weights in a batch of windows."""
    asset_wealth = (1 + returns).cumprod(dim=-2)
    wealth = (asset_wealth * weights.unsqueeze(-2)).sum(dim=-1)
    previous = torch.cat((torch.ones_like(wealth[..., :1]), wealth[..., :-1]), dim=-1)
    drifted = weights * asset_wealth[..., -1, :] / wealth[..., -1:]
    return wealth / previous - 1, drifted


def portfolio_loss(returns: torch.Tensor, weights: torch.Tensor, turnover_lambda: float):
    daily, drifted = buy_and_hold(returns, weights)
    risk = daily.var(dim=-1, unbiased=True).mean() * RISK_SCALE
    changes = weights[1:] - drifted[:-1]
    half_l1 = .5 * ((changes.square() + TURNOVER_SMOOTHING ** 2).sqrt() - TURNOVER_SMOOTHING).sum(dim=-1)
    turnover = (1. + half_l1.sum()) / weights.shape[0]
    return risk + turnover_lambda * turnover, risk, turnover


@dataclass
class FittedPortfolioRouter:
    model: PortfolioAwareMLP
    scaler: StandardScaler
    config: PortfolioConfig
    history: list[dict]
    best_epoch: int

    @torch.no_grad()
    def allocate(self, features: np.ndarray, covariances: list[np.ndarray]):
        inputs = torch.as_tensor(self.scaler.transform(features[None]), dtype=torch.float64)
        covs = torch.as_tensor(np.stack(covariances), dtype=torch.float64)
        weights, expert_weights = self.model(inputs[0], covs)
        return weights.numpy(), expert_weights.numpy()


def _training_tensors(samples, scaler):
    features = scaler.transform(np.stack([sample.features for sample in samples]))
    return tuple(torch.as_tensor(values, dtype=torch.float64) for values in (
        features, np.stack([sample.covs for sample in samples]),
        np.stack([sample.next_returns for sample in samples]),
    ))


def fit_portfolio_router(train, validation, config: PortfolioConfig) -> FittedPortfolioRouter:
    if config.max_epochs < 1 or config.patience < 1:
        raise ValueError("max_epochs and patience must be positive.")
    if not math.isfinite(config.turnover_lambda) or config.turnover_lambda < 0:
        raise ValueError("Turnover lambda must be non-negative and finite.")
    for split in (train, validation):
        if not split or any(left.end != right.start for left, right in zip(split, split[1:])):
            raise ValueError("Portfolio training requires consecutive chronological windows within each split.")
    if train[-1].end > validation[0].start:
        raise ValueError("Training and validation holding periods must be disjoint and chronological.")

    scaler = StandardScaler().fit(np.stack([sample.features for sample in train]))
    x_train, cov_train, r_train = _training_tensors(train, scaler)
    x_val, cov_val, r_val = _training_tensors(validation, scaler)
    model = PortfolioAwareMLP(x_train.shape[-1], cov_train.shape[-1], cov_train.shape[1], config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history = []
    best_loss, best_epoch = math.inf, 0
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        weights, _ = model(x_train, cov_train)
        loss, risk, turnover = portfolio_loss(r_train, weights, config.turnover_lambda)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training portfolio loss at epoch {epoch}.")
        loss.backward()
        gradients = [p.grad for p in model.network.parameters()]
        if any(gradient is None or not torch.isfinite(gradient).all() for gradient in gradients):
            raise RuntimeError(f"Missing or non-finite portfolio gradient at epoch {epoch}.")
        norm = torch.linalg.vector_norm(torch.cat([gradient.flatten() for gradient in gradients])).item()
        if norm == 0:
            raise RuntimeError(f"Portfolio gradient is zero at epoch {epoch}.")
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_weights, _ = model(x_val, cov_val)
            val_loss, val_risk, val_turnover = portfolio_loss(r_val, val_weights, config.turnover_lambda)
        value = val_loss.item()
        if not math.isfinite(value):
            raise RuntimeError(f"Non-finite validation portfolio loss at epoch {epoch}.")
        history.append({"epoch": epoch, "train_loss": loss.item(), "train_risk": risk.item(),
                        "train_turnover": turnover.item(), "gradient_norm": norm,
                        "validation_loss": value, "validation_risk": val_risk.item(),
                        "validation_turnover": val_turnover.item()})
        if value < best_loss:
            best_loss, best_epoch = value, epoch
            best_state = deepcopy(model.network.state_dict())
        if epoch - best_epoch >= config.patience:
            break

    model.network.load_state_dict(best_state)
    return FittedPortfolioRouter(model, scaler, config, history, best_epoch)

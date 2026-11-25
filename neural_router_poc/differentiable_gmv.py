"""Differentiable CPU long-only GMV, with the reference solver's covariance scale."""
import math

import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer
from torch import nn


COVARIANCE_SCALE = 1e4
DEFAULT_JITTER = 1e-9  # In daily-return covariance units, before scaling.
FEASIBILITY_TOL = 1e-6
SOLVER_ARGS = {"solve_method": "Clarabel", "tol_gap_abs": 1e-10,
               "tol_gap_rel": 1e-10, "tol_feas": 1e-10, "max_iter": 300,
               "n_jobs_forward": 1}


def covariance_mixture(scores: torch.Tensor, experts: torch.Tensor, temperature: float):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Softmax temperature must be positive and finite.")
    probabilities = torch.softmax(-scores / temperature, dim=-1)
    covariance = torch.einsum("...e,...eij->...ij", probabilities, experts)
    return covariance, probabilities


class DifferentiableGMV(nn.Module):
    def __init__(self, assets: int, jitter: float = DEFAULT_JITTER):
        super().__init__()
        if not math.isfinite(jitter) or jitter <= 0:
            raise ValueError("GMV diagonal jitter must be positive and finite.")
        self.jitter = jitter
        self.register_buffer("identity", torch.eye(assets, dtype=torch.float64))
        factor = cp.Parameter((assets, assets))
        weights = cp.Variable(assets)
        problem = cp.Problem(cp.Minimize(cp.sum_squares(factor @ weights)),
                             [cp.sum(weights) == 1, weights >= 0, weights <= 1])
        if not problem.is_dpp():
            raise RuntimeError("The factor-parameterized GMV problem must be DPP compliant.")
        self.layer = CvxpyLayer(problem, parameters=[factor], variables=[weights],
                                solver=cp.DIFFCP, solver_args=SOLVER_ARGS)

    def forward(self, covariance: torch.Tensor) -> torch.Tensor:
        symmetric = (covariance + covariance.transpose(-1, -2)) * .5
        scaled = (symmetric + self.jitter * self.identity) * COVARIANCE_SCALE
        # torch.linalg.cholesky gives L L.T. CVXPY needs A.T A, so A = L.T.
        factor = torch.linalg.cholesky(scaled).transpose(-1, -2)
        # DIFFCP's inference-only entry point accepts forward worker settings only.
        (weights,) = self.layer(factor, solver_args={"n_jobs_backward": 1} if factor.requires_grad else None)
        if (not torch.isfinite(weights).all() or torch.any(weights < -FEASIBILITY_TOL)
                or torch.any(weights > 1 + FEASIBILITY_TOL)
                or torch.any((weights.sum(dim=-1) - 1).abs() > FEASIBILITY_TOL)):
            raise RuntimeError("Differentiable GMV returned an infeasible solution.")
        weights = weights.clamp_min(0)
        return weights / weights.sum(dim=-1, keepdim=True)

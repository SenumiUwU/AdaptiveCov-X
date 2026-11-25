import importlib.util
import unittest

import numpy as np


HAS_LAYER = importlib.util.find_spec("cvxpylayers") is not None


@unittest.skipUnless(HAS_LAYER, "Install requirements-portfolio.txt for differentiable GMV tests")
class DifferentiableGMVTests(unittest.TestCase):
    def test_forward_and_backward_agree_with_reference_and_finite_differences(self):
        import torch
        from differentiable_gmv import DifferentiableGMV, covariance_mixture
        from portfolio_aware_router import portfolio_loss
        from run_poc import gmv_weights

        assets = 9
        rng = np.random.default_rng(259)
        experts = torch.tensor(np.stack([np.diag(rng.uniform(.6e-4, 2e-4, assets)) for _ in range(3)]))
        returns = torch.tensor(rng.normal(0., .013, (1, 21, assets)))
        scores = torch.tensor([.2, -.1, .35], dtype=torch.float64, requires_grad=True)
        gmv = DifferentiableGMV(assets)

        def solve(current_scores):
            cov, probs = covariance_mixture(current_scores, experts, .75)
            weights = gmv(cov)
            loss, _, _ = portfolio_loss(returns, weights.unsqueeze(0), 0.)
            return cov, probs, weights, loss

        cov, probs, weights, loss = solve(scores)
        loss.backward()
        self.assertTrue((weights >= 0.).all() and (weights <= 1.).all())
        self.assertTrue((probs >= 0.).all())
        self.assertAlmostEqual(probs.sum().item(), 1.)
        self.assertAlmostEqual(weights.sum().item(), 1.)
        self.assertGreater(torch.linalg.eigvalsh(cov).min().item(), 0.)
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertGreater(scores.grad.norm().item(), 1e-4)
        with torch.no_grad():
            reference = gmv_weights(cov.numpy() + gmv.jitter * np.eye(assets))
            np.testing.assert_allclose(weights.numpy(), reference, atol=2e-5)
            step = 1e-3 * torch.eye(3, dtype=scores.dtype)
            numerical = torch.stack([(solve(scores + delta)[-1] - solve(scores - delta)[-1]) / .002
                                     for delta in step])
        torch.testing.assert_close(scores.grad, numerical, rtol=.03, atol=.004)

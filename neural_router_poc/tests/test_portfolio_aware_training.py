import importlib.util
import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from paired_bootstrap import paired_volatility_bootstrap
from run_poc import _hold_returns, build_samples


HAS_LAYER = importlib.util.find_spec("cvxpylayers") is not None


@unittest.skipUnless(HAS_LAYER, "Install requirements-portfolio.txt for portfolio training tests")
class PortfolioTrainingTests(unittest.TestCase):
    def test_buy_and_hold_and_turnover_match_drifted_holdings(self):
        import torch
        from portfolio_aware_router import RISK_SCALE, buy_and_hold, portfolio_loss

        returns = torch.tensor([[[.12, -.05], [.08, -.03]], [[-.11, .09], [-.04, .05]]], dtype=torch.float64)
        weights = torch.tensor([[.5, .5], [.5, .5]], dtype=torch.float64, requires_grad=True)
        daily, drifted = buy_and_hold(returns, weights)
        for period, initial, realised, drift in zip(returns.tolist(), weights.tolist(), daily.tolist(), drifted.tolist()):
            expected, last = _hold_returns(np.array(period), np.array(initial))
            np.testing.assert_allclose(realised, expected, atol=1e-14)
            np.testing.assert_allclose(drift, last, atol=1e-14)
        loss, risk, turnover = portfolio_loss(returns, weights, 25.)
        exact = (1. + .5 * (weights[1] - drifted[0]).abs().sum().item()) / 2.
        self.assertAlmostEqual(turnover.item(), exact, places=4)
        self.assertGreater(turnover.item(), .5)
        self.assertAlmostEqual(risk.item(), np.var(daily.tolist(), axis=1, ddof=1).mean() * RISK_SCALE)
        self.assertAlmostEqual(loss.item(), risk.item() + 25 * turnover.item())
        turnover.backward()
        self.assertTrue(torch.isfinite(weights.grad).all())
        self.assertGreater(weights.grad[0].norm().item(), .1)

    def test_training_uses_portfolio_outcomes_and_returns_checkpoint(self):
        from portfolio_aware_router import PortfolioConfig, fit_portfolio_router

        rng = np.random.default_rng(123)
        dates = pd.bdate_range("2020-01-01", periods=420)
        returns = rng.normal(0, .015, (len(dates), 9))
        samples = build_samples(dates, returns, rng.normal(0, .01, len(dates)), rng.uniform(15, 30, len(dates)))
        train, val = samples[:4], samples[4:6]
        config = PortfolioConfig(max_epochs=3, patience=2, turnover_lambda=25)
        fitted = fit_portfolio_router(train, val, config)
        # Expert log-variance labels are irrelevant to end-to-end portfolio training.
        replaced = [replace(sample, target_log_vars=np.full(3, np.nan)) for sample in train]
        repeat = fit_portfolio_router(replaced, val, config)
        self.assertEqual(fitted.history, repeat.history)
        self.assertEqual(fitted.best_epoch, min(fitted.history, key=lambda row: row["validation_loss"])["epoch"])
        self.assertTrue(all(row["gradient_norm"] > 0 for row in fitted.history))
        np.testing.assert_allclose(fitted.scaler.mean_, np.mean([s.features for s in train], axis=0))
        hold = samples[6]
        changed = replace(hold, next_returns=np.full_like(hold.next_returns, .11))
        weights, experts = fitted.allocate(hold.features, hold.covs)
        np.testing.assert_array_equal(weights, fitted.allocate(changed.features, changed.covs)[0])
        self.assertTrue((weights >= 0).all() and (experts >= 0).all())
        self.assertAlmostEqual(weights.sum(), 1.)
        self.assertAlmostEqual(experts.sum(), 1.)


class BootstrapTests(unittest.TestCase):
    def test_bootstrap_preserves_pairing_and_checks_alignment(self):
        x = np.random.default_rng(13).normal(0, .01, 126)
        data = pd.DataFrame({"a": x, "b": x}, index=pd.bdate_range("2020-01-01", periods=len(x)))
        result = paired_volatility_bootstrap(data, "a", "b", resamples=200)
        self.assertEqual(result["annualized_volatility_difference_bps"], 0.)
        self.assertEqual(result["ci95_bps"], [0., 0.])
        data.b = 2 * x
        bigger = paired_volatility_bootstrap(data, "b", "a", resamples=200)
        self.assertAlmostEqual(bigger["annualized_volatility_difference_bps"], x.std(ddof=1) * np.sqrt(252) * 1e4)
        self.assertTrue(bigger["ci_excludes_zero"])
        with self.assertRaisesRegex(ValueError, "increasing trading dates"):
            paired_volatility_bootstrap(data.iloc[::-1], "b", "a", resamples=200)

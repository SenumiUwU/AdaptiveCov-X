"""Run from PoC directory with: python -m unittest discover -s tests -v"""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from neural_router import fit_neural_router
from run_poc import (COST_BPS, EXPERTS, SECTORS, _hold_returns, backtest,
                     build_samples, covariance_experts, fit_router, gmv_weights,
                     load_prices, make_synthetic_csv, router_probs, split_samples,
                     weights_for)

class PoCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        csv = Path(cls.temp.name) / "synthetic.csv"
        make_synthetic_csv(csv, days=1500, seed=13)
        cls.dates, cls.returns, cls.spy, cls.vix = load_prices(csv)
        cls.samples = build_samples(cls.dates, cls.returns, cls.spy, cls.vix)
        cls.train, cls.validation, cls.test = split_samples(cls.samples)
        cls.x_train = np.stack([s.features for s in cls.train])
        cls.x_val = np.stack([s.features for s in cls.validation])
        cls.y_val = np.stack([s.target_log_vars for s in cls.validation])
        cls.neural = fit_neural_router(
            cls.x_train, np.stack([s.target_log_vars for s in cls.train]),
            cls.x_val, cls.y_val, seed=42, learning_rate=.003, weight_decay=.01,
            max_epochs=100, patience=15,
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_no_temporal_leakage(self):
        label_rows = [{i for s in split for i in range(s.start, s.end)}
                      for split in (self.train, self.validation, self.test)]
        for early, late in zip(label_rows, label_rows[1:]):
            self.assertLess(max(early), min(late))
            self.assertTrue(early.isdisjoint(late))
        self.assertEqual(self.samples[0].start, 252)
        self.assertEqual(self.samples[0].end, 273)
        self.assertEqual(len(self.samples[0].next_returns), 21)

    def test_split_rejects_label_crossing_boundary(self):
        for boundary in (len(self.train), len(self.train) + len(self.validation)):
            with self.subTest(boundary=boundary):
                samples = self.samples.copy()
                samples[boundary - 1] = replace(samples[boundary - 1], end=samples[boundary].start + 1)
                with self.assertRaisesRegex(ValueError, "non-overlapping label windows"):
                    split_samples(samples)

    def test_future_prices_change_labels_only(self):
        sample = self.samples[0]
        returns = self.returns[:sample.end].copy()
        returns[sample.start:] *= 2
        spy = self.spy[:sample.end].copy()
        spy[sample.start:] *= 2
        vix = self.vix[:sample.end].copy()
        vix[sample.start:] += 10
        recomputed, = build_samples(self.dates[:sample.end], returns, spy, vix)
        np.testing.assert_array_equal(sample.features, recomputed.features)
        np.testing.assert_array_equal(sample.covs, recomputed.covs)
        np.testing.assert_array_equal(sample.expert_weights, recomputed.expert_weights)
        self.assertFalse(np.allclose(sample.target_log_vars, recomputed.target_log_vars))

    def test_experts_are_positive_semidefinite_and_gmv_long_only(self):
        for c in covariance_experts(self.returns[:252]):
            self.assertGreaterEqual(float(np.linalg.eigvalsh(c).min()), -1e-9)
            w = gmv_weights(c)
            self.assertAlmostEqual(w.sum(), 1, places=7)
            self.assertTrue((w >= -1e-8).all())

    def test_router_output_is_simplex_and_blend_psd(self):
        sample = self.test[0]
        for name, model in (("Ridge", fit_router(self.train, alpha=10)), ("Neural MLP", self.neural)):
            with self.subTest(strategy=name):
                portfolio, probs = weights_for(name, sample, model, temperature=.75)
                self.assertEqual(probs.shape, (len(EXPERTS),))
                self.assertAlmostEqual(probs.sum(), 1, places=8)
                self.assertTrue((probs >= 0).all())
                cov = sum(weight * expert_cov for weight, expert_cov in zip(probs, sample.covs))
                self.assertEqual(cov.shape, (len(SECTORS), len(SECTORS)))
                np.testing.assert_allclose(cov, cov.T, atol=1e-12)
                self.assertGreaterEqual(np.linalg.eigvalsh(cov).min(), -1e-9)
                np.testing.assert_allclose(portfolio, gmv_weights(cov), atol=1e-10)
                self.assertTrue((portfolio >= 0).all())
                self.assertAlmostEqual(portfolio.sum(), 1, places=8)

    def test_softmax_prefers_lower_risk_and_requires_positive_temperature(self):
        model = SimpleNamespace(predict=lambda _: np.array([[-10., -8., -9.]]))
        sharp = router_probs(model, self.test[0], temperature=.35)
        flat = router_probs(model, self.test[0], temperature=1.5)
        self.assertGreater(sharp[0], sharp[2])
        self.assertGreater(sharp[2], sharp[1])
        self.assertGreater(sharp[0], flat[0])
        for invalid in (0., -1., np.inf, np.nan):
            with self.subTest(temperature=invalid), self.assertRaisesRegex(ValueError, "temperature"):
                router_probs(model, self.test[0], temperature=invalid)

    def test_neural_router_shapes_and_training_only_scaler(self):
        self.assertEqual(self.x_train.shape[1], 56)
        self.assertEqual(self.neural.predict(self.x_train[:1]).shape, (1, len(EXPERTS)))
        scores = self.neural.predict(self.x_val)
        self.assertEqual(scores.shape, self.y_val.shape)
        self.assertTrue(np.isfinite(scores).all())
        np.testing.assert_allclose(self.neural.scaler.mean_, self.x_train.mean(axis=0))
        self.assertEqual(self.neural.scaler.n_samples_seen_, len(self.train))
        self.assertAlmostEqual(np.mean((scores - self.y_val) ** 2), self.neural.validation_mse, places=6)

    def test_neural_early_stopping_respects_patience(self):
        # With zero learning rate, validation MSE stops improving after epoch 1.
        model = fit_neural_router(
            self.x_train, np.stack([s.target_log_vars for s in self.train]),
            self.x_val, self.y_val, seed=42, learning_rate=0., weight_decay=.01,
            max_epochs=10, patience=3,
        )
        self.assertEqual(model.best_epoch, 1)
        self.assertEqual(model.epochs_trained, 4)

    def test_backtests_share_schedule_and_cost(self):
        models = {"Ridge": fit_router(self.train, alpha=10), "Neural MLP": self.neural}
        samples = self.test[:2]
        for strategy in (*EXPERTS, "Static Blend", "Equal Weight", *models):
            with self.subTest(strategy=strategy):
                daily, records, _ = backtest(samples, strategy, models.get(strategy))
                np.testing.assert_array_equal(daily.Index, np.arange(samples[0].start, samples[-1].end))
                self.assertEqual(records.Date.tolist(), [str(s.date.date()) for s in samples])
                w = records[[f"Portfolio_{ticker}" for ticker in SECTORS]].to_numpy(dtype=float)
                self.assertTrue((w >= 0).all())
                np.testing.assert_allclose(w.sum(axis=1), 1.)
                for sample, target, turnover in zip(samples, w, records.Turnover):
                    gross_first = target @ sample.next_returns[0]
                    net_first = (1. - turnover * COST_BPS / 1e4) * (1. + gross_first) - 1.
                    self.assertAlmostEqual(daily.loc[daily.Index == sample.start, "DailyReturn"].item(), net_first)
                self.assertEqual(records.Turnover.iloc[0], 1.)

    def test_buy_and_hold_and_nonzero_features(self):
        s = self.samples[4]
        out, last_weights = _hold_returns(s.next_returns, np.full(9, 1/9))
        self.assertEqual(len(out), 21)
        self.assertAlmostEqual(last_weights.sum(), 1, places=7)
        self.assertTrue(np.isfinite(s.features).all())


class YahooSourceTests(unittest.TestCase):
    """Offline parser contract tests: NO claim of live Yahoo download."""

    def _fake_yahoo(self):
        import pandas as pd
        from download_real_data import TICKERS
        days = pd.bdate_range(end="2026-11-11", periods=1150)
        cols = pd.MultiIndex.from_product([["Close", "Volume"], TICKERS], names=["Price", "Ticker"])
        x = np.ones((len(days), len(cols)), dtype=float) * 100
        for i, tick in enumerate(TICKERS):
            x[:, i] = 100 + .05 * np.arange(len(days)) + i
        x[-3, 3] = np.nan  # one isolated gap, visible in audit
        return pd.DataFrame(x, index=days, columns=cols)

    def test_yahoo_column_format_gap_audit_fixed_cutoff(self):
        from download_real_data import clean_yahoo_frame, TICKERS
        close, audit = clean_yahoo_frame(self._fake_yahoo(), "2022-01-01", "2026-11-12")
        self.assertEqual(list(close.columns), [*TICKERS[:-1], "VIX"])
        self.assertEqual(str(close.index[-1].date()), "2026-11-11")
        self.assertFalse(close.isna().any().any())
        self.assertEqual(int(audit.loc[audit.Ticker == "XLI", "Rows_missing_before_drop"].iloc[0]), 1)
        self.assertEqual(len(close), 1149)

    def test_yahoo_parser_rejects_missing_ticker_and_invalid_coverage(self):
        from download_real_data import clean_yahoo_frame
        frame = self._fake_yahoo().drop(columns=[("Close", "XLB")])
        with self.assertRaisesRegex(ValueError, "cannot locate|did not return|symbols"):
            clean_yahoo_frame(frame, "2022-01-01", "2026-11-12")
        with self.assertRaisesRegex(ValueError, "complete price rows"):
            clean_yahoo_frame(self._fake_yahoo(), "2022-01-01", "2026-11-12", min_rows=2000)


if __name__ == "__main__":
    unittest.main()

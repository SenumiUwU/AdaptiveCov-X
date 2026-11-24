"""Run from PoC directory with: python -m unittest discover -s tests -v"""
import tempfile
import unittest
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd

from neural_router import fit_neural_router
from run_poc import (COST_BPS, EXPERTS, FEATURE_NAMES, LOOKBACK, SECTORS,
                     _hold_returns, _vix_5d_change, backtest,
                     build_samples, covariance_experts, fit_router, gmv_weights,
                     load_prices, make_synthetic_csv, router_probs, split_samples,
                     weights_for)

class PoCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.csv = Path(cls.temp.name) / "synthetic.csv"
        make_synthetic_csv(cls.csv, days=1500, seed=13)
        cls.prices = load_prices(cls.csv)
        cls.samples = build_samples(*cls.prices)
        cls.legacy = build_samples(*cls.prices, feature_set="legacy56")
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
        frame = pd.read_csv(self.csv, parse_dates=["Date"])
        future = frame.Date >= sample.date
        frame.loc[future, [*SECTORS, "SPY"]] *= 1.025
        frame.loc[future, "VIX"] += 10
        path = self.csv.with_name("future_changed.csv")
        frame.to_csv(path, index=False, float_format="%.8f")
        history_and_hold = tuple(values[:sample.end] for values in load_prices(path))
        for feature_set, original in (("expanded58", sample), ("legacy56", self.legacy[0])):
            with self.subTest(feature_set=feature_set):
                recomputed, = build_samples(*history_and_hold, feature_set=feature_set)
                np.testing.assert_array_equal(original.features, recomputed.features)
                np.testing.assert_array_equal(original.covs, recomputed.covs)
                np.testing.assert_array_equal(original.expert_weights, recomputed.expert_weights)
                self.assertFalse(np.allclose(original.target_log_vars, recomputed.target_log_vars))

    def test_lagged_market_features_and_sector_scaling(self):
        t = LOOKBACK
        dates, returns, _, _ = (values[:t + 21] for values in self.prices)
        # Use business-day rows spanning a weekend and a distinct calendar-day value.
        self.assertEqual((dates[t - 1] - dates[t - 6]).days, 7)
        calendar_date = dates[t - 1].date() - timedelta(days=5)
        calendar_offset = dates.get_loc(pd.Timestamp(calendar_date))
        self.assertNotEqual(calendar_offset, t - 6)
        vix = np.full(len(dates), 15.)
        vix[[t - 6, calendar_offset, t - 1, t]] = [20., 25., 30., 40.]
        spy = np.zeros(len(dates))
        spy[t - 22] = .40
        spy[t - 21:t - 1] = .01
        spy[t - 1] = -.03
        spy[t:] = .50
        names = FEATURE_NAMES["expanded58"]
        expected_new = {"vix_5d_change": 30 / 20 - 1, "spy_21d_return": 1.01 ** 20 * .97 - 1}
        expanded, = build_samples(dates, returns, spy, vix)
        self.assertEqual(names[:4], ("vix_lagged_level", "vix_5d_change",
                                     "spy_21d_annualized_volatility", "spy_21d_return"))
        self.assertEqual(expanded.features[0], 30.)
        self.assertAlmostEqual(expanded.features[2], np.std([.01] * 20 + [-.03], ddof=1) * np.sqrt(252))
        for feature_set, schema in FEATURE_NAMES.items():
            with self.subTest(feature_set=feature_set):
                sample, = build_samples(dates, returns, spy, vix, feature_set=feature_set)
                self.assertEqual(sample.features.shape, (int(feature_set[-2:]),))
                self.assertTrue(np.isfinite(sample.features).all())
                np.testing.assert_array_equal(sample.features, [expanded.features[names.index(n)] for n in schema])
                for name, value in expected_new.items():
                    if name in schema:
                        self.assertAlmostEqual(sample.features[schema.index(name)], value)
        recent = returns[t - 21:t]
        row, col = np.triu_indices(len(SECTORS))
        np.testing.assert_allclose(expanded.features[4:13], np.std(recent, axis=0, ddof=1) * 100)
        np.testing.assert_allclose(expanded.features[13:], np.cov(recent.T, ddof=1)[row, col] * 1e4)

    def test_vix_change_rejects_invalid_history(self):
        with self.assertRaisesRegex(ValueError, "six observations"):
            _vix_5d_change(np.ones(5), 5)
        for invalid in (0., -1., np.inf, np.nan):
            with self.subTest(denominator=invalid), self.assertRaisesRegex(ValueError, "denominator"):
                _vix_5d_change(np.array([invalid, 20., 20., 20., 20., 25.]), 6)

    def test_spy_features_reject_incomplete_history(self):
        dates, returns, spy, vix = self.prices
        with self.assertRaisesRegex(ValueError, "SPY features require 21 observations"):
            build_samples(dates, returns, spy[:LOOKBACK - 1], vix)

    def test_experts_are_positive_semidefinite_and_gmv_long_only(self):
        for c in covariance_experts(self.prices[1][:LOOKBACK]):
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

    def test_both_routers_accept_each_schema_and_fit_training_only_scalers(self):
        for feature_set, schema in FEATURE_NAMES.items():
            with self.subTest(feature_set=feature_set):
                train, validation, test = split_samples(build_samples(*self.prices, feature_set=feature_set))
                x_train = np.stack([s.features for s in train])
                x_val = np.stack([s.features for s in validation])
                neural = fit_neural_router(
                    x_train, np.stack([s.target_log_vars for s in train]), x_val, self.y_val,
                    seed=42, learning_rate=.003, weight_decay=.01, max_epochs=30, patience=5,
                )
                ridge = fit_router(train, alpha=10)
                self.assertEqual(neural.network[0].in_features, len(schema))
                self.assertEqual(neural.network[-1].out_features, len(EXPERTS))
                for model, scaler in ((ridge, ridge.named_steps["standardscaler"]), (neural, neural.scaler)):
                    scores = model.predict(x_val)
                    self.assertEqual(scores.shape, self.y_val.shape)
                    self.assertTrue(np.isfinite(scores).all())
                    np.testing.assert_allclose(scaler.mean_, x_train.mean(axis=0))
                    self.assertEqual(scaler.n_samples_seen_, len(train))
                    probs = router_probs(model, test[0], temperature=.75)
                    self.assertTrue((probs >= 0).all())
                    self.assertAlmostEqual(probs.sum(), 1.)
                self.assertAlmostEqual(np.mean((neural.predict(x_val) - self.y_val) ** 2),
                                       neural.validation_mse, places=6)

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
        expanded_windows = [(s.date, s.start, s.end) for s in self.samples]
        self.assertEqual([(s.date, s.start, s.end) for s in self.legacy], expanded_windows)
        for legacy, expanded in zip(self.legacy, self.samples):
            np.testing.assert_array_equal(legacy.covs, expanded.covs)
            np.testing.assert_array_equal(legacy.target_log_vars, expanded.target_log_vars)
        legacy_train, legacy_val, legacy_test = split_samples(self.legacy)
        legacy_neural = fit_neural_router(
            np.stack([s.features for s in legacy_train]), np.stack([s.target_log_vars for s in legacy_train]),
            np.stack([s.features for s in legacy_val]), self.y_val,
            seed=42, learning_rate=.003, weight_decay=.01, max_epochs=30, patience=5,
        )
        self.assertEqual(COST_BPS, 10)
        for train, samples, neural in ((self.train, self.test, self.neural), (legacy_train, legacy_test, legacy_neural)):
            models = {"Ridge": fit_router(train, alpha=10), "Neural MLP": neural}
            for strategy in (*EXPERTS, "Static Blend", "Equal Weight", *models):
                with self.subTest(strategy=strategy, width=len(samples[0].features)):
                    daily, records, total_turnover = backtest(samples, strategy, models.get(strategy))
                    np.testing.assert_array_equal(daily.Index, np.arange(self.test[0].start, self.test[-1].end))
                    self.assertEqual(records.Date.tolist(), [str(s.date.date()) for s in self.test])
                    w = records[[f"Portfolio_{ticker}" for ticker in SECTORS]].to_numpy(dtype=float)
                    self.assertTrue((w >= 0).all())
                    np.testing.assert_allclose(w.sum(axis=1), 1.)
                    self.assertAlmostEqual(total_turnover, records.Turnover.sum())
                    last_weights = None
                    for sample, target, turnover in zip(samples, w, records.Turnover):
                        expected_turnover = 1. if last_weights is None else .5 * np.abs(target - last_weights).sum()
                        self.assertAlmostEqual(turnover, expected_turnover)
                        gross, last_weights = _hold_returns(sample.next_returns, target)
                        gross[0] = (1. - turnover * COST_BPS / 1e4) * (1. + gross[0]) - 1.
                        period = daily.loc[daily.Index.between(sample.start, sample.end - 1), "DailyReturn"]
                        np.testing.assert_allclose(period, gross)

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

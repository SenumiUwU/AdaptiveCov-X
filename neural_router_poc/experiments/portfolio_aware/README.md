# Portfolio-aware routing on the audited Yahoo snapshot

**This holdout provides insufficient evidence that end-to-end training materially reduces volatility.** The risk-only router reduced annualised net volatility by 1.50 bps versus the supervised 58-input MLP. The paired 95% interval was −10.74 to +8.76 bps and the point estimate fell short of the pre-specified 10 bp reduction threshold. The turnover-penalised model increased held-out turnover.

## Data and controls

The source was the Phase 1 Yahoo Finance snapshot `data/real_prices.csv`, SHA-256 `5c88bf34f6897e86258b65f2e845270d835b05afabf1e4c5d9b791faaf829b87`. It contains 3,235 aligned price rows from 2014-01-02 through 2026-11-11. The chronological split contains 85 training, 28 validation, and 29 test rebalances. The held-out daily returns run from 2024-06-10 through 2026-11-11, for 609 sessions. All strategies use the same 252-session covariance lookback, 21-session holdings, long-only, fully invested portfolios, and 10 bps per unit of one-way turnover. Existing strategies that solve GMV use SLSQP. The portfolio-aware strategies use CVXPYLayers at training and inference.

## Results

| Strategy | Net ann. volatility | Net ann. return | Net Sharpe | Max drawdown | Turnover |
| --- | ---: | ---: | ---: | ---: | ---: |
| Supervised MLP (56) | 11.0348% | 7.9091% | 0.7451 | −12.0212% | 4.2973 |
| Supervised MLP (58) | 11.0613% | 8.1490% | 0.7637 | −12.2260% | 5.0514 |
| Portfolio-Aware MLP (58) | 11.0463% | 8.2799% | 0.7755 | −11.8680% | 3.9756 |
| Portfolio-Aware MLP + Turnover (58) | 11.0460% | 8.2485% | 0.7729 | −11.9818% | 4.4160 |
| Ledoit–Wolf | 10.9620% | 10.4834% | 0.9645 | −11.1347% | 2.5202 |

- **A, features.** The 58-input supervised model reduced validation log-variance MSE from 0.2311 to 0.2125, while holdout volatility increased by 2.65 bps. [Experiment A](experiment_a_features.csv)
- **B, training objective.** The risk-only model reduced turnover from 5.05 to 3.98 relative to supervised 58. Its mean Historical, EWMA, and Ledoit–Wolf weights were 54.1%, 21.2%, and 24.7%, versus 30.0%, 31.8%, and 38.2% for the supervised model. Mean portfolio Herfindahl concentration rose from 0.3083 to 0.3105. Ledoit–Wolf on its own had the lowest volatility among the fixed and learned comparators in B. [Experiment B](experiment_b_objective.csv)
- **C, turnover.** At the selected checkpoint, the turnover-aware model improved the penalised validation objective from 187.81 to 186.39 in annualised percentage-points squared. The average smooth turnover component was slightly higher, 0.1429 versus 0.1423. On the holdout, exact turnover increased to 4.42 and volatility changed by −0.02 bps. [Experiment C](experiment_c_turnover.csv)
- **D, robustness.** All six one-factor trials finished. Held-out volatility ranged from 11.0454% to 11.0732%, and turnover ranged from 3.98 to 5.27. Changing jitter from `1e-9` to `1e-7` at the reference temperature and seed changed net volatility by about 0.03 bps. Seeds 13 and 73 were sensitivity checks outside the validation selection. [Experiment D](experiment_d_sensitivity.csv)

The paired moving-block bootstrap resamples the same days for both strategies. Its 21-session blocks reflect the holding period and retain some within-hold dependence. It uses 10,000 resamples with seed 42. The percentile intervals are conditional on the fitted strategies and this holdout.

| Comparison, alternative minus reference | Difference, bps | 95% interval, bps |
| --- | ---: | ---: |
| Supervised 58 minus supervised 56 | +2.65 | [−8.02, +13.44] |
| Portfolio-aware risk minus supervised 58 | −1.50 | [−10.74, +8.76] |
| Portfolio-aware turnover minus portfolio-aware risk | −0.02 | [−5.39, +5.84] |

All three intervals include zero and offer insufficient evidence for a 10 bp volatility reduction. The 58-input architectures share layer widths, while the supervised output bias is initialised from the training log-variance labels. The risk-only synthetic smoke run was 3.87 bps more volatile than supervised 58 on its simulated holdout. [Bootstrap results](paired_bootstrap.json) · [Paired net daily returns](test_daily_returns.csv) · [Rebalance allocations](test_rebalance_weights.csv)

## Configuration and numerical checks

AdamW used learning rate `.003`, weight decay `.01`, 60 maximum epochs, and patience 12 for portfolio training. Validation selected seed 42, temperature `.35`, and daily covariance jitter `1e-9` from the eligible risk trials. Risk-only selected epoch 1 and trained through epoch 13. Turnover used the matched setting, penalty coefficient 25, and selected epoch 55 of 60. The training objective and its units are described in the [PoC README](../../README.md). [Selected parameters, split dates, warnings, and epoch histories](experiment_metadata.json)

All 19 tests passed. A non-degenerate nine-asset case verifies score gradients against central finite differences, with a separate training test verifying finite, nonzero network gradients. An audit checked 493 exported portfolios and recomputed their buy-and-hold daily returns and trading costs from the same 609 sessions. Across the portfolio-aware allocations, the largest absolute weight difference from SLSQP on the same jittered objective was `1.12e-5`. Both supervised feature sets reproduced their previous standalone metrics, daily returns, and allocations byte for byte.

During development, the DIFFCP solve-only call rejected the derivative worker argument with `AttributeError` and the first sensitivity evaluation rejected trial aliases with `ValueError`. The final reruns resolved both. CVXPYLayers emitted NumPy `__array__` copy-keyword deprecation warnings and the Yahoo parser test emitted an existing NumPy timedelta deprecation warning. The archived runs finished without reported solver convergence or feasibility failures.

## Executed commands

Commands were run from `neural_router_poc/` on Windows with Python 3.13.2, PyTorch 2.15.0 on CPU, NumPy 2.5.5, pandas 2.3.3, SciPy 1.18.1, scikit-learn 1.9.1, CVXPY 1.9.3, CVXPYLayers 1.2.0, DIFFCP 1.1.9, and Clarabel 0.11.1. The initial install built DIFFCP from source. Install the optional requirements with `python -m pip install -r requirements-portfolio.txt` in a clean environment.

```powershell
.\.venv\Scripts\python.exe -m pip --disable-pip-version-check --retries 1 --timeout 20 install --progress-bar off 'cvxpy>=1.7,<2' 'cvxpylayers>=1.0,<2'
.\.venv\Scripts\python.exe -m pip --disable-pip-version-check install --no-index -r requirements-portfolio.txt
.\.venv\Scripts\python.exe -m pip check
$env:OPENBLAS_NUM_THREADS='1'; $env:OMP_NUM_THREADS='1'; $env:PYTHONUNBUFFERED='1'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe run_poc.py --csv data/real_prices.csv --training-objective all --out results_portfolio_yahoo
.\.venv\Scripts\python.exe run_poc.py --demo --training-objective portfolio-risk --out results_portfolio_synthetic
.\.venv\Scripts\python.exe paired_bootstrap.py results_portfolio_yahoo/test_daily_returns.csv --alternative 'Portfolio-Aware MLP' --reference 'Neural MLP' --block-length 21 --resamples 10000 --seed 42 --materiality-bps 10
```

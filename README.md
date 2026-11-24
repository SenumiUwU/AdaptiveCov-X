# AdaptiveCov-X

AdaptiveCov-X is a research and implementation project focused on adaptive covariance estimation and portfolio optimization.

## Original notebook implementation

[`notebooks/01_data_collection.ipynb`](notebooks/01_data_collection.ipynb) contains the earlier Yahoo Finance data collection, covariance diagnostics, minimum-variance portfolios, and exploratory rule-based and neural routing experiments. The original raw data is under `data/raw/`, and the saved outputs are under `notebooks/`.

## Ridge and PyTorch routing proof of concept

[`neural_router_poc/`](neural_router_poc/) runs a standalone experiment on nine US sector ETFs. Ridge and a compact PyTorch MLP (58 → 16 → 8 → 3) predict the next 21 trading days' realised log variance for the Historical, EWMA, and Ledoit–Wolf experts. The default `expanded58` feature set includes five-trading-day VIX change and 21-trading-day cumulative SPY return. `--feature-set legacy56` uses the original inputs for both routers. Softmax turns the negative forecasts into covariance blend weights. A separate long-only global minimum variance optimiser computes the portfolio. The seven-strategy comparison uses a 252-trading-day covariance lookback, 21-trading-day rebalancing, and a 10 bps one-way transaction cost.

### Run the PoC

The PoC was tested with Python 3.13 and CPU PyTorch. Its [requirements](neural_router_poc/requirements.txt) also include NumPy, pandas, SciPy, scikit-learn, Matplotlib, and yfinance. Create a virtual environment from the repository root:

```powershell
cd neural_router_poc
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe run_poc.py --demo
.\.venv\Scripts\python.exe run_poc.py --yahoo
```

On macOS or Linux, use `python3 -m venv .venv` and `./.venv/bin/python` for the same commands. The synthetic demo writes `results_synthetic/viva_dashboard.html`. A successful Yahoo run writes `results_yahoo/viva_dashboard.html` and an audited price snapshot under `neural_router_poc/data/`. To reuse the snapshot, run `run_poc.py --csv data/real_prices.csv` with the same Python interpreter.

Add `--ablation` to compare the 56-, 57-, and 58-feature variants on a single snapshot. See the [feature-ablation report](neural_router_poc/experiments/feature_ablation_report.md) for the Yahoo Finance and synthetic results.

### Current limitations

The routers are supervised risk forecasters, with portfolio optimisation downstream. The experiment uses one chronological holdout per dataset, a fixed ETF universe, and the assumption that fills occur at the preceding close. Treat synthetic results as pipeline verification and historical results as one empirical comparison. Yahoo availability and revisions can affect reproducibility. See the [PoC documentation](neural_router_poc/README.md) for the split, input schema, and outputs.

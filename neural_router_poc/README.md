# Covariance routing PoC

Run commands from this directory using the Python environment in the [setup instructions](../README.md#run-the-poc).

## Models and evaluation

Both routers default to `expanded58`. For the first holding return at index `t`, the inputs are ordered as follows. Indices count aligned trading observations.

1. VIX level at `t-1`.
2. Five-day VIX percentage change, `vix[t-1] / vix[t-6] - 1`.
3. Annualised 21-day SPY sample volatility, `std(spy[t-21:t], ddof=1) * sqrt(252)`.
4. Compounded SPY simple return over the same 21 observations, `prod(1 + spy[t-21:t]) - 1`.
5. Nine sector daily sample volatilities from the prior 21 observations, multiplied by 100.
6. The 45 upper-triangular entries of the 21-day sector sample covariance matrix, including the diagonal in row-major order, multiplied by 10,000.

`--feature-set legacy56` omits both added indicators. `--feature-set vix57` adds only VIX change, and `--feature-set spy57` adds only SPY return. The MLP uses the selected input width, 16- and 8-unit ReLU layers, and three outputs. Both routers predict the natural logs of the sample variances of each expert's next 21 daily gross buy-and-hold GMV returns. The variance floor is `1e-12`.

The 60/20/20 training, validation, and test split is chronological and uses disjoint label windows. Scaling and parameter updates use the training windows. The MLP uses AdamW with learning rate `0.003`, weight decay `0.01`, and at most 300 epochs. Validation log-variance MSE selects the checkpoint with patience of 25 epochs. Validation net portfolio volatility selects the Ridge alpha and each router's temperature. All feature sets use the same windows, expert covariances, and targets. On test, both routers share the 252-day lookback, 21-day rebalances, and 10 bps one-way cost convention with Historical, EWMA, Ledoit–Wolf, Static Blend, and Equal Weight. Turnover is half the L1 distance from the previous drifted weights, with turnover of one at initial investment.

## Price inputs

- `--demo` generates deterministic synthetic prices for an offline run.
- `--yahoo` downloads nine sector ETFs, SPY, and the VIX index level via yfinance. The default request starts on 2014-01-01 and ends on 2026-11-12 (exclusive). The downloader writes the cleaned CSV, its SHA-256, and per-ticker gap counts to `data/`.
- `--csv PATH` reads a single-header file with `Date,XLB,XLE,XLF,XLI,XLK,XLP,XLU,XLV,XLY,SPY,VIX`. Use adjusted closes for the ETFs and index levels for VIX. The original repository's raw Yahoo CSV has a multi-row header and requires conversion to this schema before use.

## Outputs

Each run writes `test_metrics.csv`, `test_daily_returns.csv`, `test_rebalance_weights.csv`, `run_metadata.json`, and `viva_dashboard.html`. Metadata records the ordered feature names, source CSV SHA-256, split dates, training configuration, and validation selections. Open the HTML file in a browser to see the comparison and both learned expert-weight series. Pass `--out results_custom` to keep a run separate.

```powershell
python run_poc.py --demo --ablation
python run_poc.py --csv data/real_prices.csv --ablation
```

`--ablation` runs all four feature sets on the same loaded snapshot. It writes a comparison CSV and metadata JSON at the output root, and the full backtest outputs in a subdirectory per feature set. The [feature-ablation report](experiments/feature_ablation_report.md) links to the archived comparison and configurations. Keep `yahoo_download_metadata.json` alongside the source CSV to retain its provenance when rerunning the study.

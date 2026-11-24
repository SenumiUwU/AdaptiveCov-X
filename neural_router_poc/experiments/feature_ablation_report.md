# Market-state feature ablation

The Yahoo Finance holdout gives an inconclusive result for the two added indicators. The 58-feature routers forecast validation log variance more accurately, but both had higher held-out turnover and higher net volatility, the primary portfolio metric. The MLP also had a higher Sharpe ratio.

## Yahoo Finance holdout

The audited snapshot contains 3,235 complete price rows from 2014-01-02 through 2026-11-11. The chronological split has 85 training, 28 validation, and 29 test rebalances, with test returns from 2024-06-10 through 2026-11-11. The data hash and every split date are recorded in the [Yahoo experiment metadata](feature_ablation_yahoo_metadata.json).

| Router | Features | Validation MSE | Net ann. vol. | Net Sharpe (rf=0) | Max drawdown | Total turnover |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Ridge | Legacy 56 | 0.4483 | 11.0593% | 0.7401 | −12.2663% | 4.7042 |
| Ridge | VIX change only | 0.4312 | 11.0647% | 0.7406 | −12.2859% | 4.7810 |
| Ridge | SPY return only | 0.4072 | 11.0599% | 0.7405 | −12.2766% | 4.7173 |
| Ridge | Expanded 58 | 0.3908 | 11.0640% | 0.7407 | −12.2882% | 4.7776 |
| MLP | Legacy 56 | 0.2311 | 11.0348% | 0.7451 | −12.0212% | 4.2973 |
| MLP | VIX change only | 0.2059 | 11.0702% | 0.7315 | −12.3451% | 4.9601 |
| MLP | SPY return only | 0.2084 | 11.0823% | 0.7200 | −12.6452% | 5.6375 |
| MLP | Expanded 58 | 0.2125 | 11.0613% | 0.7637 | −12.2260% | 5.0514 |

From 56 to 58 features, held-out volatility rose by **0.47 bps for Ridge** and **2.65 bps for the MLP**. Held-out prediction MSE also rose for both. SPY return alone left Ridge volatility within 0.06 bps of legacy and produced the best held-out Ridge forecast MSE across the four variants. Both single-indicator MLPs had higher volatility, lower Sharpe ratios, and deeper drawdowns than the legacy MLP.

Ridge mean expert allocations moved by less than 0.3 percentage points per expert from 56 to 58 features. The MLP shifted mean weight from Ledoit–Wolf (41.2% to 38.2%) toward EWMA (28.1% to 31.8%). Its Ledoit–Wolf weight standard deviation fell from 14.45 to 5.23 percentage points. The VIX-only MLP selected a temperature of 1.5 and averaged roughly one-third per expert. The SPY-only MLP averaged 40.5% in EWMA. The [comparison CSV](feature_ablation_comparison.csv) contains forecast MSE, net returns and risk metrics, and mean and standard deviation of test expert weights for every run.

## Controls and synthetic check

Every feature set used the same dates, covariances, targets, seed 42, and portfolio constraints within each dataset. Covariance lookback was 252 trading days, rebalancing was every 21 trading days, and costs were 10 bps per unit of one-way turnover. The MLP checkpoint was selected on validation prediction MSE. Validation net volatility selected Ridge alpha and both softmax temperatures, using the original grids. Test metrics were calculated after those selections.

The synthetic demonstration completed all four configurations, with 64 training, 21 validation, and 22 test rebalances. From 56 to 58 features, net annualised volatility moved from 12.2989% to 12.2946% for Ridge and from 12.2886% to 12.3305% for the MLP. These runs verify the pipeline on simulated prices. See the [synthetic experiment metadata](feature_ablation_synthetic_metadata.json).

The Yahoo results describe one holdout and one MLP seed. Changing the input width also changes the MLP's initial weights under the same seeded initialisation, so the observed differences include sensitivity to that initialisation.

## Reproduce

From `neural_router_poc/`, use an environment with the project requirements installed and the original Yahoo CSV and audit JSON in `data/`. The archived metadata records the SHA-256 to verify the exact snapshot. The runs used CPU PyTorch and the thread settings below.

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python -m unittest discover -s tests -v
python run_poc.py --demo --ablation
python run_poc.py --csv data/real_prices.csv --ablation
```

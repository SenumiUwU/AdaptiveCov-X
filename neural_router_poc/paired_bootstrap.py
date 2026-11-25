"""Paired moving-block bootstrap of held-out annualised volatility differences."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def paired_volatility_bootstrap(returns: pd.DataFrame, alternative: str, reference: str,
                                *, block_length: int = 21, resamples: int = 10000,
                                seed: int = 42, materiality_bps: float = 10.):
    if not returns.index.is_unique or not returns.index.is_monotonic_increasing:
        raise ValueError("Paired returns require unique, increasing trading dates.")
    values = returns[[alternative, reference]].to_numpy(dtype=float)
    n = len(values)
    if alternative == reference or n < 3 or not np.isfinite(values).all():
        raise ValueError("Two distinct strategies with complete paired daily returns are required.")
    if not 1 <= block_length < n or resamples < 2:
        raise ValueError("Require 1 <= block length < number of days and at least two resamples.")
    if not np.isfinite(materiality_bps) or materiality_bps < 0:
        raise ValueError("Materiality must be non-negative and finite.")

    # Non-circular overlapping source blocks, paired across both strategies.
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block_length + 1, (resamples, (n + block_length - 1) // block_length))
    scale = np.sqrt(252) * 1e4
    differences = np.empty(resamples)
    for start in range(0, resamples, 256):
        indices = (starts[start:start + 256, :, None] + np.arange(block_length)).reshape(-1, starts.shape[1] * block_length)
        vols = values[indices[:, :n]].std(axis=1, ddof=1) * scale
        differences[start:start + len(vols)] = vols[:, 0] - vols[:, 1]
    point = float(np.diff(values.std(axis=0, ddof=1))[0] * -scale)
    low, high = np.quantile(differences, [.025, .975])
    return {"alternative": alternative, "reference": reference, "days": n,
            "annualized_volatility_difference_bps": point, "ci95_bps": [float(low), float(high)],
            "ci_excludes_zero": bool(high < 0 or low > 0), "materiality_threshold_bps": materiality_bps,
            "point_meets_improvement_threshold": point <= -materiality_bps,
            "ci_supports_material_improvement": bool(high <= -materiality_bps),
            "block_length": block_length, "resamples": resamples, "seed": seed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("returns", type=Path, help="CSV with aligned net daily returns and a Date column")
    parser.add_argument("--alternative", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--block-length", type=int, default=21)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--materiality-bps", type=float, default=10.)
    args = parser.parse_args()
    returns = pd.read_csv(args.returns, parse_dates=["Date"], index_col="Date")
    result = paired_volatility_bootstrap(returns, args.alternative, args.reference,
                                        block_length=args.block_length, resamples=args.resamples,
                                        seed=args.seed, materiality_bps=args.materiality_bps)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

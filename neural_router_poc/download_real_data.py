"""Download reproducible daily Yahoo Finance adjusted ETF closes + VIX levels.

Default snapshot ends BEFORE 2026-11-12, so the U.S. trading day of
12 November cannot enter the study before its close (Sri Lanka is ahead).
The endpoint is exclusive and the last requested trading day is 2026-11-11.

Usage:
  python download_real_data.py
  python download_real_data.py --start 2014-01-01 --end-exclusive 2026-11-12

The function is also called by `python run_poc.py --yahoo`.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
TICKERS = [*SECTORS, "SPY", "^VIX"]
START = "2014-01-01"
END_EXCLUSIVE = "2026-11-12"


def clean_yahoo_frame(raw: pd.DataFrame, start: str, end_exclusive: str,
                      min_rows: int = 1001) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select auto-adjusted Close prices, preserve ^VIX as a *level*.

    Reject incomplete fields and record how many gaps were dropped. No filling
    across non-trading days or interpolation is performed.
    """
    if raw is None or raw.empty:
        raise ValueError("Yahoo Finance returned zero rows. Data not saved.")
    if not isinstance(raw.columns, pd.MultiIndex):
        raise ValueError("Expected MultiIndex Yahoo columns for 11 tickers, but received a single-level frame.")
    levels = [list(map(str, raw.columns.get_level_values(i))) for i in range(raw.columns.nlevels)]
    matching = [i for i, names in enumerate(levels) if "Close" in names]
    if len(matching) != 1:
        raise ValueError("Cannot locate the Yahoo Close field; inspect the returned MultiIndex.")
    close = raw.xs("Close", axis=1, level=matching[0]).copy()
    close.columns = close.columns.map(str)
    absent = sorted(set(TICKERS) - set(close.columns))
    if absent:
        raise ValueError(f"Yahoo Finance did not return these symbols: {absent}")
    close = close[TICKERS].copy()
    close.index = pd.to_datetime(close.index)
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)
    close.index = close.index.normalize()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    close = close.loc[(close.index >= pd.Timestamp(start)) &
                      (close.index < pd.Timestamp(end_exclusive))]
    close = close.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    for ticker in TICKERS:
        close.loc[close[ticker] <= 0, ticker] = np.nan
    gap_counts = close.isna().sum()
    quality = pd.DataFrame({
        "Ticker": TICKERS,
        "Role": ["sector ETF" for _ in SECTORS] + ["market ETF", "VIX index level"],
        "Rows_missing_before_drop": [int(gap_counts[k]) for k in TICKERS],
        "First_available": [str(close[k].first_valid_index().date()) if close[k].first_valid_index() is not None else "" for k in TICKERS],
        "Last_available": [str(close[k].last_valid_index().date()) if close[k].last_valid_index() is not None else "" for k in TICKERS],
    })
    complete = close.dropna(how="any").rename(columns={"^VIX": "VIX"})
    if len(complete) < min_rows:
        raise ValueError(f"Only {len(complete)} complete price rows; require at least {min_rows} (the 252/21 window backtest needs a substantial history).")
    if complete.index[-1] < pd.Timestamp(end_exclusive) - pd.Timedelta(days=12):
        raise ValueError("Latest complete row is >12 calendar days before the cutoff. Check Yahoo freshness and missing symbols.")
    if not (complete.index.to_series().diff().dropna().dt.days <= 8).all():
        raise ValueError("Found a gap >8 calendar days between complete daily rows; investigate before use.")
    if complete.index.has_duplicates or complete.isna().any().any() or (complete <= 0).any().any():
        raise ValueError("Dataset contains duplicate dates, nonpositive values or missing values.")
    complete.index.name = "Date"
    return complete, quality


def download_yahoo(start: str = START, end_exclusive: str = END_EXCLUSIVE,
                   output: Path = Path("data/real_prices.csv")) -> tuple[Path, dict]:
    if pd.Timestamp(end_exclusive) <= pd.Timestamp(start):
        raise ValueError("End-exclusive date must be later than start.")
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install the Yahoo downloader first: python -m pip install -r requirements.txt") from exc
    print(f"Downloading daily Yahoo Finance data for {', '.join(TICKERS)}")
    print(f"Range: {start} (inclusive) to {end_exclusive} (exclusive); no unclosed U.S. session")
    try:
        raw = yf.download(
            tickers=TICKERS, start=start, end=end_exclusive, interval="1d",
            auto_adjust=True, group_by="column", progress=False,
            actions=False, threads=False, timeout=30,
        )
    except Exception as exc:
        raise RuntimeError("Yahoo Finance download failed. Verify internet access, rate limits or remote availability; no synthetic data was substituted.") from exc
    complete, quality = clean_yahoo_frame(raw, start, end_exclusive)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # A complete real-data snapshot with no interpolation or forward fill.
    complete.to_csv(output, float_format="%.12g", date_format="%Y-%m-%d")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    meta = {
        "data_kind": "Yahoo Finance historical data",
        "source": "Yahoo Finance market data via yfinance (unofficial third-party package)",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "yfinance_version": getattr(yf, "__version__", "unknown"),
        "tickers": TICKERS,
        "adjusted_etf_prices": True,
        "vix_is_index_level_not_total_return": True,
        "requested_start_inclusive": start,
        "requested_end_exclusive": end_exclusive,
        "actual_first_date": str(complete.index[0].date()),
        "actual_last_date": str(complete.index[-1].date()),
        "complete_price_rows": len(complete),
        "raw_rows_in_requested_window": len(raw),
        "rows_removed_by_complete_case_alignment": int(len(raw) - len(complete)),
        "missing_values_interpolated": False,
        "data_sha256": digest,
        "csv_path": str(output),
    }
    # Keep audit files alongside prices, without changing the user's existing notebook data.
    quality.to_csv(output.parent / "yahoo_download_quality.csv", index=False)
    (output.parent / "yahoo_download_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved {len(complete)} adjusted-price rows from {meta['actual_first_date']} to {meta['actual_last_date']}: {output}")
    print("Data SHA-256:", digest)
    print("ETF closes are dividend/split adjusted; VIX is a raw index LEVEL used as a lagged feature.")
    return output, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default=START)
    ap.add_argument("--end-exclusive", default=END_EXCLUSIVE)
    ap.add_argument("--output", type=Path, default=Path("data/real_prices.csv"))
    args = ap.parse_args()
    download_yahoo(args.start, args.end_exclusive, args.output)


if __name__ == "__main__":
    main()

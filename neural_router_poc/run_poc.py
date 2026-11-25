"""AdaptiveCov-X: small, reproducible, leakage-aware research proof of concept.

Usage: python run_poc.py --yahoo
       python run_poc.py --csv data/real_prices.csv --ablation
       python run_poc.py --demo --feature-set expanded58

The supplied demo is SYNTHETIC. It is not evidence of real-market performance.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
from dataclasses import dataclass, replace
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
EXPERTS = ["Historical", "EWMA", "Ledoit-Wolf"]
ROUTERS = ("Ridge", "Neural MLP")
PORTFOLIO_ROUTERS = ("Portfolio-Aware MLP", "Portfolio-Aware MLP + Turnover")
LOOKBACK, REBALANCE, TRADING_DAYS, COST_BPS = 252, 21, 252, 10
MARKET_WINDOW, VIX_CHANGE_DAYS = 21, 5
EWMA_LAMBDA = 0.94
STATIC_ROUTER = np.array([0.35, 0.20, 0.45])
COLORS = ["#2864a1", "#d69a27", "#298c78", "#8554ad", "#cb5f4e", "#7d8795", "#131d33"]
DEFAULT_FEATURE_SET = "expanded58"
UPPER_TRIANGLE = np.triu_indices(len(SECTORS))
EXPANDED_FEATURE_NAMES = (
    "vix_lagged_level", "vix_5d_change", "spy_21d_annualized_volatility", "spy_21d_return",
    *(f"{ticker}_21d_daily_volatility_pct" for ticker in SECTORS),
    *(f"{SECTORS[i]}_{SECTORS[j]}_21d_covariance_x10000" for i, j in zip(*UPPER_TRIANGLE)),
)
_ABLATION_EXCLUSIONS = {
    "legacy56": {"vix_5d_change", "spy_21d_return"},
    "vix57": {"spy_21d_return"},
    "spy57": {"vix_5d_change"},
    "expanded58": set(),
}
FEATURE_NAMES = {
    name: tuple(feature for feature in EXPANDED_FEATURE_NAMES if feature not in excluded)
    for name, excluded in _ABLATION_EXCLUSIONS.items()
}
_FEATURE_COLUMNS = {
    name: [EXPANDED_FEATURE_NAMES.index(feature) for feature in features]
    for name, features in FEATURE_NAMES.items()
}


@dataclass
class Sample:
    start: int
    end: int
    date: pd.Timestamp
    features: np.ndarray
    covs: list[np.ndarray]
    expert_weights: np.ndarray
    target_log_vars: np.ndarray
    next_returns: np.ndarray
    vix: float
    spy_vol: float


def make_synthetic_csv(path: Path, days: int = 2520, seed: int = 42) -> None:
    """Deterministic nine-sector market with changing common-factor volatility.
    Synthetic SPY and VIX are generated from the same simulated state.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-04", periods=days)
    # Cyclical and stochastic high-correlation, high-volatility regimes.
    regime = np.zeros(days, dtype=bool)
    regime[280:450] = True
    regime[780:950] = True
    regime[1350:1610] = True
    regime[1950:2110] = True
    common = rng.standard_normal(days)
    idiosyn = rng.standard_normal((days, len(SECTORS)))
    betas = np.array([0.83, 1.35, 1.04, 0.97, 1.17, 0.56, 0.42, 0.58, 1.00])
    sector_vols = np.array([.008, .015, .011, .009, .012, .006, .007, .007, .011])
    mkt_vol = np.where(regime, .026, .008)
    common_factor = common * mkt_vol
    shock = idiosyn * sector_vols * np.where(regime[:, None], 1.45, 1.0)
    r = .00022 + common_factor[:, None] * betas + shock
    r = np.clip(r, -.22, .22)
    prices = 100 * np.cumprod(1 + r, axis=0)
    spy_r = .0002 + .75 * common_factor + .20 * r.mean(axis=1) + rng.normal(0, .003, days)
    spy = 100 * np.cumprod(1 + np.clip(spy_r, -.22, .22))
    vix = np.clip(np.where(regime, 34.0, 15.0) + rng.normal(0, 3.1, days), 9, 80)
    df = pd.DataFrame(prices, columns=SECTORS)
    df.insert(0, "Date", dates)
    df["SPY"], df["VIX"] = spy, vix
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.8f")


def load_prices(path: Path | io.BytesIO) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path, parse_dates=["Date"])
    required = ["Date", *SECTORS, "SPY", "VIX"]
    missing = [x for x in required if x not in df.columns]
    if missing:
        raise ValueError(f"CSV needs these columns: {missing}. VIX is a volatility index level, NOT a price.")
    df = df[required].copy().drop_duplicates("Date").sort_values("Date").dropna()
    if (df[[*SECTORS, "SPY"]].to_numpy(dtype=float) <= 0).any():
        raise ValueError("All sector ETF and SPY price values must be positive adjusted close prices.")
    if (df["VIX"].to_numpy(dtype=float) <= 0).any():
        raise ValueError("VIX level must be positive.")
    r_df = df[[*SECTORS, "SPY"]].pct_change().iloc[1:]
    if not np.isfinite(r_df.to_numpy()).all():
        raise ValueError("Non-finite daily returns: check data continuity and prices.")
    if len(r_df) < 1000:
        raise ValueError("At least 1000 daily return observations are needed for this PoC (5+ years recommended).")
    return (pd.DatetimeIndex(df["Date"].iloc[1:]),
            r_df[SECTORS].to_numpy(dtype=float),
            r_df["SPY"].to_numpy(dtype=float),
            df["VIX"].iloc[1:].to_numpy(dtype=float))


def nearest_psd(cov: np.ndarray) -> np.ndarray:
    cov = (cov + cov.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    return (eigenvectors * np.maximum(eigenvalues, 1e-10)) @ eigenvectors.T


def covariance_experts(window: np.ndarray) -> list[np.ndarray]:
    hist = np.cov(window, rowvar=False, ddof=1)
    # Exponential weights: newest observation carries the most weight.
    n = len(window)
    weights = (1 - EWMA_LAMBDA) * EWMA_LAMBDA ** np.arange(n - 1, -1, -1)
    weights /= weights.sum()
    mu = np.sum(window * weights[:, None], axis=0)
    x = window - mu
    ewma = x.T @ (x * weights[:, None])
    lw = LedoitWolf().fit(window).covariance_
    return [nearest_psd(c) for c in (hist, ewma, lw)]


def gmv_weights(cov: np.ndarray) -> np.ndarray:
    """Long-only, fully invested minimum variance portfolio (scaled objective)."""
    sigma = nearest_psd(cov) * 1e4
    n = sigma.shape[0]
    w0 = np.full(n, 1 / n)
    result = minimize(lambda w: float(w @ sigma @ w), w0,
                      jac=lambda w: 2 * sigma @ w,
                      constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1,
                                    "jac": lambda w: np.ones_like(w)}],
                      bounds=[(0, 1)] * n, method="SLSQP",
                      options={"maxiter": 1000, "ftol": 1e-10})
    if not result.success or not np.isfinite(result.x).all():
        # Defensible fallback in case of numerical optimisation failure.
        raise RuntimeError(f"SLSQP failed: {result.message}")
    w = np.clip(result.x, 0, 1)
    return w / w.sum()


def _hold_returns(period: np.ndarray, starting_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Buy-and-hold between rebalances: weights drift with asset prices."""
    w = starting_weights.copy()
    out = np.zeros(len(period))
    for j, daily_assets in enumerate(period):
        gross = float(w @ daily_assets)
        out[j] = gross
        w = w * (1 + daily_assets) / (1 + gross)
    return out, w


def _vix_5d_change(vix: np.ndarray, t: int) -> float:
    if t <= VIX_CHANGE_DAYS or t > len(vix):
        raise ValueError("VIX change requires six observations before the rebalance.")
    denominator = float(vix[t - 1 - VIX_CHANGE_DAYS])
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("VIX change denominator must be a positive, finite index level.")
    return float(vix[t - 1]) / denominator - 1.0


def build_samples(dates: pd.DatetimeIndex, r: np.ndarray, spy: np.ndarray, vix: np.ndarray,
                  feature_set: str = DEFAULT_FEATURE_SET) -> list[Sample]:
    samples = []
    columns = _FEATURE_COLUMNS[feature_set]
    for t in range(LOOKBACK, len(r) - REBALANCE + 1, REBALANCE):
        lookback = r[t - LOOKBACK:t]
        recent = r[t - MARKET_WINDOW:t]
        recent_spy = spy[t - MARKET_WINDOW:t]
        if len(recent_spy) != MARKET_WINDOW:
            raise ValueError(f"SPY features require {MARKET_WINDOW} observations before the rebalance.")
        recent_cov = np.cov(recent, rowvar=False, ddof=1)
        spy_vol = float(np.std(recent_spy, ddof=1) * np.sqrt(TRADING_DAYS))
        spy_return = np.prod(1.0 + recent_spy) - 1.0
        daily_vols = np.std(recent, axis=0, ddof=1)
        vix_change = _vix_5d_change(vix, t)
        expanded = np.r_[vix[t - 1], vix_change, spy_vol, spy_return, daily_vols * 100,
                         recent_cov[UPPER_TRIANGLE] * 1e4]
        if not np.isfinite(expanded).all():
            raise ValueError(f"Non-finite market features at rebalance {dates[t].date()}.")
        feature = expanded[columns]
        covs = covariance_experts(lookback)
        expert_w = np.stack([gmv_weights(c) for c in covs])
        future = r[t:t + REBALANCE]
        # Targets use future ONLY as historical training labels, never decision features.
        variances = [np.var(_hold_returns(future, w)[0], ddof=1) for w in expert_w]
        samples.append(Sample(t, t + REBALANCE, dates[t], feature, covs, expert_w,
                              np.log(np.maximum(variances, 1e-12)), future,
                              float(vix[t - 1]), spy_vol))
    return samples


def split_samples(samples: list[Sample]):
    n = len(samples)
    a, b = int(.6 * n), int(.8 * n)
    if a < 15 or b - a < 5 or n - b < 5:
        raise ValueError("Not enough rebalances for chronological train/validation/test splits.")
    if any(left.end > right.start or left.date >= right.date
           for left, right in zip(samples, samples[1:])):
        raise ValueError("Rebalance samples must be chronological with non-overlapping label windows.")
    return samples[:a], samples[a:b], samples[b:]


def fit_router(train: list[Sample], alpha: float):
    features = np.stack([s.features for s in train])
    targets = np.stack([s.target_log_vars for s in train])
    # Scaler is fitted on TRAIN ONLY; multioutput Ridge predicts three log risks.
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(features, targets)
    return model


def router_probs(model, sample: Sample, temperature: float) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Softmax temperature must be positive and finite.")
    scores = np.asarray(model.predict(sample.features[None, :])[0], dtype=float)
    if scores.shape != (len(EXPERTS),) or not np.isfinite(scores).all():
        raise ValueError("Router must predict one finite log variance per covariance expert.")
    return softmax((scores.min() - scores) / temperature)


def weights_for(strategy: str, sample: Sample, model=None, temperature: float = 1.):
    if strategy in PORTFOLIO_ROUTERS:
        return model.allocate(sample.features, sample.covs)
    if strategy == "Equal Weight":
        return np.full(len(SECTORS), 1 / len(SECTORS)), np.array([np.nan] * 3)
    if strategy in EXPERTS:
        k = EXPERTS.index(strategy)
        probs = np.eye(3)[k]
        return sample.expert_weights[k], probs
    if strategy == "Static Blend":
        probs = STATIC_ROUTER
    elif strategy == "Rule-Based":
        probs = (np.array([.15, .45, .40]) if (sample.vix >= 25 or sample.spy_vol >= .25)
                 else STATIC_ROUTER)
    elif strategy in (*ROUTERS, "AdaptiveCov-X"):
        if model is None:
            raise ValueError(f"{strategy} requires a trained router")
        probs = router_probs(model, sample, temperature)
    else:
        raise ValueError(strategy)
    cov = sum(float(p) * c for p, c in zip(probs, sample.covs))
    return gmv_weights(cov), probs


def backtest(samples: list[Sample], strategy: str, model=None, temperature: float = 1.):
    rows, records = [], []
    last_w = None
    total_turnover = 0.
    initial_date = samples[0].date
    for sample in samples:
        w, probs = weights_for(strategy, sample, model, temperature)
        # 0.5 * L1 is one-way portfolio turnover. Initial investment from cash = 1.
        turnover = 1. if last_w is None else float(.5 * np.abs(w - last_w).sum())
        cost = turnover * COST_BPS / 1e4
        gross, last_w = _hold_returns(sample.next_returns, w)
        net = gross.copy()
        net[0] = (1. - cost) * (1. + gross[0]) - 1.
        total_turnover += turnover
        records.append({"Date": str(sample.date.date()), "Strategy": strategy,
                        "Turnover": turnover,
                        **{f"Weight_{e}": float(probs[i]) for i, e in enumerate(EXPERTS)},
                        **{f"Portfolio_{ticker}": float(w[i]) for i, ticker in enumerate(SECTORS)}})
        for d, value in zip(range(sample.start, sample.end), net):
            rows.append((d, float(value)))
    return pd.DataFrame(rows, columns=["Index", "DailyReturn"]), pd.DataFrame(records), total_turnover


def risk_metrics(daily: np.ndarray, total_turnover: float):
    wealth = np.cumprod(1 + daily)
    running_max = np.maximum.accumulate(np.r_[1., wealth])[1:]
    ann_vol = float(np.std(daily, ddof=1) * np.sqrt(TRADING_DAYS))
    return {"Ann. return": float(wealth[-1] ** (TRADING_DAYS / len(daily)) - 1),
            "Ann. volatility": ann_vol,
            "Sharpe (rf=0)": float(np.mean(daily) / np.std(daily, ddof=1) * np.sqrt(TRADING_DAYS)) if ann_vol > 0 else np.nan,
            "Max drawdown": float(np.min(wealth / running_max - 1)),
            "Total turnover": float(total_turnover)}


def evaluate(samples: list[Sample], strategy: str, model=None, temperature: float = 1.):
    daily, weights, turnover = backtest(samples, strategy, model, temperature)
    return risk_metrics(daily.DailyReturn.to_numpy(), turnover), daily, weights


def chart_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def make_report(output: Path, metrics: pd.DataFrame, daily_returns: pd.DataFrame,
                weights: pd.DataFrame, meta: dict, source_is_synthetic: bool):
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(10.4, 4.7))
    for i, name in enumerate(metrics.index):
        curve = (1 + daily_returns[name]).cumprod()
        ax.plot(daily_returns.index, curve, lw=2 if name in ROUTERS else 1.4,
                label=name, color=COLORS[i % len(COLORS)])
    ax.set_title("Held-out test: net cumulative wealth (starting at 1.00)", loc="left", fontsize=13, fontweight="bold")
    ax.set_ylabel("Growth of 1.00")
    ax.grid(alpha=.2)
    ax.legend(ncol=4, fontsize=8, frameon=False)
    png_wealth = chart_base64(fig)

    fig, ax = plt.subplots(figsize=(10.4, 3.8))
    s = metrics["Ann. volatility"].sort_values()
    ax.barh(s.index, s.values * 100,
            color=[COLORS[metrics.index.get_loc(name) % len(COLORS)] for name in s.index])
    ax.invert_yaxis()
    ax.set_xlabel("Annualised net volatility (%) — lower is better")
    ax.set_title("Held-out test: realised net portfolio risk", loc="left", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=.2)
    png_risk = chart_base64(fig)

    fig, axes = plt.subplots(nrows=len(ROUTERS), sharex=True, figsize=(10.4, 5.6), layout="constrained")
    for ax, name in zip(axes, ROUTERS):
        router = weights[weights.Strategy == name]
        ax.stackplot(router.Date, *[router[f"Weight_{e}"].to_numpy() for e in EXPERTS],
                     labels=EXPERTS, colors=COLORS[:len(EXPERTS)])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Expert weight")
        ax.set_title(name, loc="left", fontsize=13, fontweight="bold")
    step = max(1, len(router) // 7)
    axes[-1].set_xticks(range(0, len(router), step), router.Date.iloc[::step], rotation=20, fontsize=8)
    axes[0].legend(loc="upper right", frameon=False, ncol=3, fontsize=8)
    png_router = chart_base64(fig)

    visible = metrics.copy()
    for c in ["Ann. return", "Ann. volatility", "Max drawdown"]:
        visible[c] = (100 * visible[c]).map(lambda v: f"{v:+.2f}%" if c == "Ann. return" else f"{v:.2f}%")
    visible["Sharpe (rf=0)"] = visible["Sharpe (rf=0)"].map(lambda v: f"{v:.3f}")
    visible["Total turnover"] = visible["Total turnover"].map(lambda v: f"{v:.2f}")
    table = visible.to_html(classes="metrics", border=0)
    warning = ("SYNTHETIC DATA — simulated market demonstration."
               if source_is_synthetic else
               "YAHOO FINANCE HISTORICAL DATA — one held-out backtest."
               if meta.get("yahoo_source_metadata") else
               "USER-SUPPLIED PRICE CSV — one held-out backtest.")
    markup = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>AdaptiveCov-X | Viva Proof of Concept</title><meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{{margin:0;background:#f2f6fa;color:#17273a;font-family:system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:1100px;margin:auto;padding:32px 24px 70px}}
header{{background:#14263e;color:white;padding:36px 24px;border-bottom:5px solid #37ac9c}}
h1{{margin:0;font-size:32px}}.sub{{opacity:.75;margin:8px 0 0}}.tag{{display:inline-block;margin-top:12px;padding:7px 11px;border-radius:6px;background:#f1d895;color:#453500;font-weight:700;font-size:12px}}.lead{{line-height:1.6}}.kpis{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}
.kpi,section{{background:white;border-radius:14px;padding:20px;margin-top:16px;box-shadow:0 1px 8px #cdd8e144}}.kpi b{{font-size:24px;display:block}}.kpi span{{font-size:12px;color:#617088}}h2{{font-size:19px;margin-top:4px}}img{{display:block;width:100%;max-width:100%}}table{{width:100%;border-collapse:collapse;font-size:13px;white-space:nowrap}}th,td{{text-align:right;border-bottom:1px solid #e0e7ee;padding:9px 7px}}th:first-child,td:first-child{{text-align:left}}th{{background:#f2f6fa}}.tablewrap{{overflow-x:auto}}.foot{{font-size:13px;color:#657286;line-height:1.65}}@media(max-width:650px){{.kpis{{grid-template-columns:1fr}}}}</style></head>
<body><header><div style="max-width:1100px;margin:auto"><h1>AdaptiveCov-X</h1><p class="sub">Leakage-aware mixture of covariance experts • long-only global minimum variance</p><span class="tag">{html.escape(warning)}</span></div></header>
<main><p class="lead">Ridge and the neural MLP use {meta['neural_mlp']['widths'][0]} lagged features to predict each covariance expert's realised log variance over the next 21 trading days. Softmax turns the negative forecasts into weights for the Historical, EWMA and Ledoit–Wolf covariances. SLSQP uses the blended covariance to compute the long-only, fully invested portfolio.</p>
<div class="kpis"><div class="kpi"><b>{len(meta['training_dates'])}</b><span>training rebalance windows</span></div><div class="kpi"><b>{len(meta['validation_dates'])}</b><span>validation windows</span></div><div class="kpi"><b>{len(meta['test_dates'])}</b><span>held-out test windows</span></div></div>
<section><h2>Baseline comparison — after 10 bps transaction costs</h2><div class="tablewrap">{table}</div></section>
<section><img src="data:image/png;base64,{png_wealth}" alt="Growth of one chart"></section>
<section><img src="data:image/png;base64,{png_risk}" alt="Annual volatility by strategy"></section>
<section><img src="data:image/png;base64,{png_router}" alt="Ridge and neural MLP covariance weights at each test rebalance"></section>
<section><h2>Evaluation protocol</h2><p class="foot">The chronological 60/20/20 split uses disjoint 21-day label windows. Scaling and gradient updates use training windows. Validation forecast error selects the MLP checkpoint. Validation portfolio volatility selects the Ridge alpha and both softmax temperatures. The seven strategies share the same test dates, rebalance schedule and 10 bps transaction-cost convention.</p></section>
<section><h2>Run metadata</h2><pre style="overflow:auto;font-size:12px">{html.escape(json.dumps({k:v for k,v in meta.items() if not k.endswith('_dates')},indent=2))}</pre></section></main></body></html>'''
    (output / "viva_dashboard.html").write_text(markup, encoding="utf-8")


def _load_experiment(csv_path: Path, synthetic: bool, source_metadata: dict | None):
    csv_path = Path(csv_path)
    snapshot = csv_path.read_bytes()
    digest = sha256(snapshot).hexdigest()
    if source_metadata is None and not synthetic:
        audit = csv_path.with_name("yahoo_download_metadata.json")
        if audit.exists():
            candidate = json.loads(audit.read_text(encoding="utf-8"))
            if candidate.get("data_sha256") == digest:
                source_metadata = candidate
    if source_metadata is not None and source_metadata.get("data_sha256") != digest:
        raise ValueError("Yahoo download metadata does not match the input CSV SHA-256.")
    dates, asset_returns, spy_returns, vix = load_prices(io.BytesIO(snapshot))
    samples = build_samples(dates, asset_returns, spy_returns, vix, DEFAULT_FEATURE_SET)
    data_kind = "synthetic" if synthetic else "Yahoo Finance historical data" if source_metadata else "user-supplied price CSV"
    meta = {"data_kind": data_kind, "source_csv": str(csv_path), "data_sha256": digest}
    if source_metadata:
        meta["yahoo_source_metadata"] = source_metadata
    return dates, samples, meta


def _run_feature_set(dates: pd.DatetimeIndex, samples: list[Sample], out: Path,
                     source_meta: dict, feature_set: str):
    from neural_router import HIDDEN_WIDTHS, fit_neural_router
    from torch import __version__ as torch_version

    out.mkdir(parents=True, exist_ok=True)
    if feature_set != DEFAULT_FEATURE_SET:
        columns = _FEATURE_COLUMNS[feature_set]
        samples = [replace(sample, features=sample.features[columns]) for sample in samples]
    train, validation, test = split_samples(samples)
    temperatures = (.35, .75, 1.5)
    alphas = (1., 10., 100.)
    ridge_candidates = []
    for alpha in alphas:
        model = fit_router(train, alpha)
        for temp in temperatures:
            metrics, _, _ = evaluate(validation, "Ridge", model, temp)
            ridge_candidates.append((metrics["Ann. volatility"], alpha, temp, model))
    ridge_val_vol, best_alpha, ridge_temp, ridge = min(ridge_candidates, key=lambda candidate: candidate[:3])

    val_features = np.stack([s.features for s in validation])
    val_targets = np.stack([s.target_log_vars for s in validation])
    neural_config = {"seed": 42, "learning_rate": .003, "weight_decay": .01,
                     "max_epochs": 300, "patience": 25}
    neural = fit_neural_router(
        np.stack([s.features for s in train]), np.stack([s.target_log_vars for s in train]),
        val_features, val_targets, **neural_config,
    )
    neural_candidates = []
    for temp in temperatures:
        metrics, _, _ = evaluate(validation, "Neural MLP", neural, temp)
        neural_candidates.append((metrics["Ann. volatility"], temp))
    neural_val_vol, neural_temp = min(neural_candidates)
    models = {"Ridge": (ridge, ridge_temp), "Neural MLP": (neural, neural_temp)}

    strategies = [*EXPERTS, "Static Blend", "Equal Weight", *ROUTERS]
    reports, all_daily, all_weights = [], [], []
    for strategy in strategies:
        metrics, daily, weights = evaluate(test, strategy, *models.get(strategy, (None, 1.)))
        reports.append({"Strategy": strategy, **metrics})
        daily["Strategy"] = strategy
        all_daily.append(daily)
        all_weights.append(weights)
    metrics = pd.DataFrame(reports).set_index("Strategy")
    long_returns = pd.concat(all_daily, ignore_index=True)
    wide_returns = long_returns.pivot(index="Index", columns="Strategy", values="DailyReturn")[strategies]
    # Use the matching daily date (sample.start -> date) for exported test returns.
    wide_returns.index = dates[wide_returns.index.to_numpy()]
    wide_returns.index.name = "Date"
    weights = pd.concat(all_weights, ignore_index=True)
    test_features = np.stack([s.features for s in test])
    test_targets = np.stack([s.target_log_vars for s in test])
    meta = {**source_meta, "nine_etfs": SECTORS, "experts": EXPERTS,
            "feature_set": feature_set, "feature_names": FEATURE_NAMES[feature_set],
            "lookback_days": LOOKBACK, "rebalance_days": REBALANCE,
            "cost_bps": COST_BPS, "cost_convention": "one-way on half-L1 turnover, initial turnover 1",
            "portfolio_constraints": "long-only, fully invested GMV",
            "annualization_days": TRADING_DAYS, "ewma_lambda": EWMA_LAMBDA,
            "market_window_days": MARKET_WINDOW, "vix_change_days": VIX_CHANGE_DAYS,
            "train_validation_test_split": "chronological 60/20/20",
            "ridge_alpha_grid": alphas, "softmax_temperature_grid": temperatures,
            "ridge_alpha_selected_on_validation": best_alpha,
            "softmax_temperature_selected_on_validation": ridge_temp,
            "neural_mlp": {**neural_config,
                           "optimizer": "AdamW", "loss": "MSE", "activation": "ReLU",
                           "torch_version": torch_version,
                           "widths": [train[0].features.size, *HIDDEN_WIDTHS, len(EXPERTS)],
                           "best_epoch": neural.best_epoch, "epochs_trained": neural.epochs_trained,
                           "validation_log_variance_mse": neural.validation_mse,
                           "softmax_temperature_selected_on_validation": neural_temp},
            "library_versions": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn")},
            "validation_log_variance_mse": {
                "Ridge": float(np.mean((ridge.predict(val_features) - val_targets) ** 2)),
                "Neural MLP": neural.validation_mse},
            "validation_net_annualized_volatility": {"Ridge": ridge_val_vol, "Neural MLP": neural_val_vol},
            "test_log_variance_mse": {name: float(np.mean((model.predict(test_features) - test_targets) ** 2))
                                      for name, (model, _) in models.items()},
            "first_test_date": str(test[0].date.date()),
            "last_test_date": str(dates[test[-1].end - 1].date()),
            "training_dates": [str(s.date.date()) for s in train],
            "validation_dates": [str(s.date.date()) for s in validation],
            "test_dates": [str(s.date.date()) for s in test],
            "last_label_dates": {name: str(dates[split[-1].end - 1].date())
                                 for name, split in zip(("training", "validation", "test"),
                                                        (train, validation, test))}}
    metrics.to_csv(out / "test_metrics.csv", float_format="%.9f")
    wide_returns.to_csv(out / "test_daily_returns.csv", float_format="%.9f")
    weights.to_csv(out / "test_rebalance_weights.csv", index=False, float_format="%.9f")
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    make_report(out, metrics, wide_returns, weights, meta, meta["data_kind"] == "synthetic")
    print(f"\nAdaptiveCov-X PoC COMPLETED | {meta['data_kind']} | {feature_set}")
    print("Test:", meta["first_test_date"], "to", meta["last_test_date"])
    print("Train/val/test windows:", len(train), len(validation), len(test))
    print("Selected Ridge alpha, softmax temp:", best_alpha, ridge_temp)
    print("Selected MLP epoch, softmax temp:", neural.best_epoch, neural_temp)
    print(metrics.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nOpen:", out / "viva_dashboard.html")
    return metrics, wide_returns, weights, meta


def run(csv_path: Path, out: Path, synthetic: bool, source_metadata: dict | None = None,
        *, feature_set: str = DEFAULT_FEATURE_SET):
    dates, samples, source_meta = _load_experiment(csv_path, synthetic, source_metadata)
    metrics, _, _, meta = _run_feature_set(dates, samples, out, source_meta, feature_set)
    return metrics, meta


def _run_ablation(dates: pd.DatetimeIndex, samples: list[Sample], out: Path, source_meta: dict):
    comparison, runs = [], {}
    for feature_set, names in FEATURE_NAMES.items():
        metrics, _, weights, meta = _run_feature_set(dates, samples, out / feature_set, source_meta, feature_set)
        runs[feature_set] = meta
        for strategy in ROUTERS:
            expert_weights = weights.loc[weights.Strategy == strategy, [f"Weight_{e}" for e in EXPERTS]]
            comparison.append({
                "Data kind": meta["data_kind"], "Source SHA-256": meta["data_sha256"],
                "Feature set": feature_set, "Features": len(names), "Strategy": strategy,
                "Validation log-variance MSE": meta["validation_log_variance_mse"][strategy],
                "Test log-variance MSE": meta["test_log_variance_mse"][strategy],
                **metrics.loc[strategy].to_dict(),
                **{f"Mean weight {e}": expert_weights[f"Weight_{e}"].mean() for e in EXPERTS},
                **{f"Weight std {e}": expert_weights[f"Weight_{e}"].std(ddof=0) for e in EXPERTS},
            })
    pd.DataFrame(comparison).to_csv(out / "feature_ablation_comparison.csv", index=False, float_format="%.9f")
    (out / "feature_ablation_metadata.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    inputs = ap.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--demo", action="store_true", help="Run the clearly labelled synthetic market demo")
    inputs.add_argument("--yahoo", action="store_true", help="Download real Yahoo Finance prices (requires internet + yfinance)")
    inputs.add_argument("--csv", type=Path, help="Path to REAL adjusted-close price CSV with SPY and VIX")
    features = ap.add_mutually_exclusive_group()
    features.add_argument("--feature-set", choices=FEATURE_NAMES, default=DEFAULT_FEATURE_SET,
                          help=f"Router input schema (default: {DEFAULT_FEATURE_SET})")
    features.add_argument("--ablation", action="store_true", help="Compare all four input schemas on one snapshot")
    ap.add_argument("--training-objective", choices=("supervised", "portfolio-risk", "portfolio-turnover", "all"),
                    default="supervised", help="Opt in to the 58-input differentiable GMV experiments")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--yahoo-start", default="2014-01-01")
    ap.add_argument("--yahoo-end-exclusive", default="2026-11-12",
                    help="Fixed historical cutoff, exclusive. Do not include a partially traded U.S. day.")
    args = ap.parse_args()
    if args.training_objective != "supervised" and (args.ablation or args.feature_set != DEFAULT_FEATURE_SET):
        ap.error("The portfolio-aware experiments use expanded58 and include their own supervised feature comparison.")
    source_metadata = None
    if args.demo:
        csv_path = Path("data") / "synthetic_demo.csv"
        if not csv_path.exists():
            make_synthetic_csv(csv_path)
    elif args.yahoo:
        from download_real_data import download_yahoo
        csv_path, source_metadata = download_yahoo(
            start=args.yahoo_start, end_exclusive=args.yahoo_end_exclusive,
            output=Path("data/real_prices.csv"))
    else:
        csv_path = args.csv
    dates, samples, source_meta = _load_experiment(csv_path, args.demo, source_metadata)
    default_out = Path("results_synthetic" if args.demo else
                       "results_yahoo" if "yahoo_source_metadata" in source_meta else "results_csv")
    if args.training_objective != "supervised":
        try:
            from train_portfolio_aware import run_portfolio_experiments
        except ModuleNotFoundError as exc:
            if exc.name not in {"cvxpy", "cvxpylayers", "diffcp"}:
                raise
            ap.error(f"Portfolio training needs {exc.name}. Install requirements-portfolio.txt.")
        suffix = "portfolio" if args.training_objective == "all" else args.training_objective.replace("-", "_")
        run_portfolio_experiments(dates, samples, source_meta,
                                  args.out or default_out.with_name(f"{default_out.name}_{suffix}"),
                                  args.training_objective)
    elif args.ablation:
        _run_ablation(dates, samples, args.out or default_out.with_name(default_out.name + "_ablation"), source_meta)
    else:
        _run_feature_set(dates, samples, args.out or default_out, source_meta, args.feature_set)


if __name__ == "__main__":
    main()

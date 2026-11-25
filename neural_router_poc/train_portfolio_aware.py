"""Validation-selected portfolio-aware experiments on the PoC's fixed holdout."""
from collections import Counter
from dataclasses import asdict, replace
from importlib.metadata import version
import json
from pathlib import Path
import platform
import warnings

from diffcp import SolverError
import pandas as pd

from differentiable_gmv import SOLVER_ARGS
from paired_bootstrap import paired_volatility_bootstrap
from portfolio_aware_router import PortfolioConfig, RISK_SCALE, TURNOVER_SMOOTHING, fit_portfolio_router
from run_poc import (EXPERTS, PORTFOLIO_ROUTERS, SECTORS, _run_feature_set, evaluate, split_samples)


LEGACY_MLP = "Supervised MLP (56)"
RISK_MLP, TURNOVER_MLP = PORTFOLIO_ROUTERS
TURNOVER_LAMBDA = 25.
RISK_CRITERION = "Mean within-window annualised gross variance in percentage-points squared"
TURNOVER_CRITERION = RISK_CRITERION + " plus 25 times mean smooth one-way turnover"


def _warnings(caught):
    counts = Counter(f"{item.category.__name__}: {item.message}" for item in caught)
    return dict(counts)


def _summary(strategy, metrics, weights, criterion, selected, cautions):
    portfolio = weights[[f"Portfolio_{ticker}" for ticker in SECTORS]]
    return {"Strategy": strategy, "Status": "ok", "Validation criterion": criterion,
            "Selected parameters": json.dumps(selected, sort_keys=True), **metrics,
            **{f"Mean weight {expert}": weights[f"Weight_{expert}"].mean() for expert in EXPERTS},
            **{f"Weight std {expert}": weights[f"Weight_{expert}"].std(ddof=0) for expert in EXPERTS},
            "Mean top asset weight": portfolio.max(axis=1).mean(),
            "Mean portfolio HHI": portfolio.pow(2).sum(axis=1).mean(),
            "Warnings": json.dumps(cautions, sort_keys=True)}


def _fit_candidate(config, train, validation):
    criterion = TURNOVER_CRITERION if config.turnover_lambda else RISK_CRITERION
    trial = {"config": asdict(config), "criterion": criterion}
    caught = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.simplefilter("default", DeprecationWarning)
            fitted = fit_portfolio_router(train, validation, config)
    except (SolverError, ValueError, RuntimeError, ArithmeticError) as exc:
        trial.update(status="failed", error=f"{type(exc).__name__}: {exc}", warnings=_warnings(caught))
        return trial, None
    trial.update(status="ok", best_epoch=fitted.best_epoch, history=fitted.history, warnings=_warnings(caught))
    trial["validation_loss"] = fitted.history[fitted.best_epoch - 1]["validation_loss"]
    return trial, fitted


def _evaluate_portfolio(strategy, fitted, trial, dates, test):
    allocation_strategy = TURNOVER_MLP if fitted.config.turnover_lambda else RISK_MLP
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.simplefilter("default", DeprecationWarning)
        metrics, daily, weights = evaluate(test, allocation_strategy, fitted)
    weights = weights.assign(Strategy=strategy)
    series = pd.Series(daily.DailyReturn.to_numpy(), index=dates[daily.Index], name=strategy)
    config = {**trial["config"], "best_epoch": trial["best_epoch"], "validation_loss": trial["validation_loss"]}
    cautions = dict(Counter(trial["warnings"]) + Counter(_warnings(caught)))
    return _summary(strategy, metrics, weights, trial["criterion"], config, cautions), series, weights


def _baseline_rows(metrics, weights, meta, cautions):
    rows = {}
    for name, values in metrics.iterrows():
        criterion, config = "Fixed allocation rule", {}
        if name == "Ridge":
            criterion = "Validation net volatility selects Ridge alpha and softmax temperature"
            config = {"alpha": meta["ridge_alpha_selected_on_validation"],
                      "temperature": meta["softmax_temperature_selected_on_validation"],
                      "validation_mse": meta["validation_log_variance_mse"][name]}
        if name == "Neural MLP":
            criterion = "Validation log-variance MSE selects checkpoint. Validation net volatility selects temperature."
            mlp = meta["neural_mlp"]
            config = {key: mlp[key] for key in ("best_epoch", "seed", "widths", "learning_rate", "weight_decay",
                                               "max_epochs", "patience", "validation_log_variance_mse",
                                               "softmax_temperature_selected_on_validation")}
        rows[name] = _summary(name, values.to_dict(), weights[weights.Strategy == name], criterion, config, cautions)
    return rows


def _write_report(out, meta, comparisons, bootstraps):
    split = meta["supervised_expanded58"]
    lines = ["# Portfolio-aware covariance routing", "",
             f"Snapshot `{split['data_sha256']}`. Test period {split['first_test_date']} through {split['last_test_date']}.",
             "", "| Experiment | Strategy | Net annualised volatility | Net Sharpe | Turnover | Status |",
             "| --- | --- | ---: | ---: | ---: | --- |"]
    for experiment, rows in comparisons.items():
        for row in rows:
            if row["Status"] == "ok":
                result = f"{row['Ann. volatility']:.4%} | {row['Sharpe (rf=0)']:.4f} | {row['Total turnover']:.4f}"
            else:
                result = " |  | "
            lines.append(f"| {experiment} | {row['Strategy']} | {result} | {row['Status']} |")
    lines.extend(["", "## Paired moving-block bootstrap", "",
                  "The 21-session block length reflects the holding period. Each resample uses the same trading days for both strategies.", ""])
    for result in bootstraps:
        low, high = result["ci95_bps"]
        diff = result["annualized_volatility_difference_bps"]
        threshold = result["materiality_threshold_bps"]
        conclusion = "meets" if result["point_meets_improvement_threshold"] else "falls short of"
        precision = "excludes" if result["ci_excludes_zero"] else "includes"
        material = "supports" if result["ci_supports_material_improvement"] else "provides insufficient evidence for"
        lines.append(f"- {result['alternative']} minus {result['reference']} was {diff:+.2f} bps "
                     f"(95% interval {low:+.2f} to {high:+.2f} bps, {result['resamples']:,} resamples). "
                     f"The point estimate {conclusion} the {threshold:g} bps reduction threshold. "
                     f"The interval {precision} zero and {material} a reduction of that size.")
    lines.extend(["", "See the experiment CSVs for allocations, concentration, and warnings, and `experiment_metadata.json` for configurations and epoch histories."])
    (out / "portfolio_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_portfolio_experiments(dates, samples, source_meta, out: Path, objective: str):
    out.mkdir(parents=True, exist_ok=True)
    train, validation, test = split_samples(samples)
    baseline = {}
    for feature_set in ("legacy56", "expanded58"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            metrics, daily, weights, meta = _run_feature_set(dates, samples, out / feature_set, source_meta, feature_set)
        baseline[feature_set] = (_baseline_rows(metrics, weights, meta, _warnings(caught)), daily, weights, meta)

    rows, daily, weights, meta58 = baseline["expanded58"]
    daily = daily.copy()
    legacy_rows, legacy_daily, legacy_weights, meta56 = baseline["legacy56"]
    rows[LEGACY_MLP] = {**legacy_rows["Neural MLP"], "Strategy": LEGACY_MLP}
    daily[LEGACY_MLP] = legacy_daily["Neural MLP"]
    rebalance_frames = [weights, legacy_weights.loc[legacy_weights.Strategy == "Neural MLP"].assign(Strategy=LEGACY_MLP)]

    rule, rule_daily, rule_weights = evaluate(test, "Rule-Based")
    if not dates[rule_daily.Index].equals(daily.index):
        raise RuntimeError("Rule-Based comparator has different test dates.")
    rows["Rule-Based"] = _summary("Rule-Based", rule, rule_weights, "Fixed allocation rule", {}, {})
    daily["Rule-Based"] = rule_daily.DailyReturn.to_numpy()
    rebalance_frames.append(rule_weights)

    reference = PortfolioConfig()
    configs = {"reference": reference}
    if objective == "all":
        configs.update(temp035=replace(reference, temperature=.35), temp150=replace(reference, temperature=1.5),
                       jitter1e7=replace(reference, jitter=1e-7), seed13=replace(reference, seed=13),
                       seed73=replace(reference, seed=73))
    trials, fits = {}, {}
    for key, config in configs.items():
        print(f"Training {key}: {config}", flush=True)
        trials[key], fits[key] = _fit_candidate(config, train, validation)
    eligible = [key for key in configs if configs[key].seed == reference.seed and fits[key] is not None]
    selected = min(eligible, key=lambda key: trials[key]["validation_loss"], default=None)
    turnover = objective in {"portfolio-turnover", "all"}
    if turnover and selected is not None:
        config = replace(configs[selected], turnover_lambda=TURNOVER_LAMBDA)
        print(f"Training turnover: {config}", flush=True)
        trials["turnover"], fits["turnover"] = _fit_candidate(config, train, validation)

    def add_test_row(name, key, trial):
        if key is None or fits[key] is None:
            return {"Strategy": name, "Status": "failed", "Validation criterion": trial["criterion"],
                    "Selected parameters": json.dumps(trial.get("config", {})),
                    "Warnings": json.dumps({"error": trial["error"], **trial.get("warnings", {})})}
        try:
            row, series, allocation = _evaluate_portfolio(name, fits[key], trial, dates, test)
        except (SolverError, ValueError, RuntimeError, ArithmeticError) as exc:
            trial["evaluation_error"] = f"{type(exc).__name__}: {exc}"
            return {"Strategy": name, "Status": "failed", "Validation criterion": trial["criterion"],
                    "Selected parameters": json.dumps(trial["config"]), "Warnings": trial["evaluation_error"]}
        if not series.index.equals(daily.index):
            raise RuntimeError(f"{name} has different test dates.")
        daily[name] = series
        rebalance_frames.append(allocation)
        return row

    rows[RISK_MLP] = add_test_row(RISK_MLP, selected, trials[selected] if selected else {
        "criterion": RISK_CRITERION, "error": "No validation candidate succeeded."})
    if turnover:
        key = "turnover" if "turnover" in trials else None
        rows[TURNOVER_MLP] = add_test_row(TURNOVER_MLP, key, trials[key] if key else {
            "criterion": TURNOVER_CRITERION, "error": "Risk-only configuration selection failed."})
    comparisons = {
        "experiment_a_features": [rows[name] for name in (LEGACY_MLP, "Neural MLP")],
        "experiment_b_objective": [rows[name] for name in (*EXPERTS, "Static Blend", "Equal Weight", "Rule-Based",
                                                          "Ridge", "Neural MLP", RISK_MLP)],
    }
    if turnover:
        comparisons["experiment_c_turnover"] = [rows[name] for name in PORTFOLIO_ROUTERS]
    if objective == "all":
        comparisons["experiment_d_sensitivity"] = [add_test_row(key, key, trials[key]) for key in configs]

    bootstraps = [paired_volatility_bootstrap(daily, alternative, reference)
                  for alternative, reference in (("Neural MLP", LEGACY_MLP), (RISK_MLP, "Neural MLP"),
                                                 (TURNOVER_MLP, RISK_MLP))
                  if alternative in daily and reference in daily]
    metadata = {"training_objective": objective, "supervised_legacy56": meta56, "supervised_expanded58": meta58,
                "selected_risk_candidate": selected, "portfolio_trials": trials,
                "solver": {"backend": "CVXPYLayers / diffcp / Clarabel", "options": SOLVER_ARGS,
                           "backward_workers": 1,
                           "risk_scale": RISK_SCALE, "turnover_smoothing": TURNOVER_SMOOTHING},
                "environment": {"python": platform.python_version(), **{name: version(name) for name in
                                ("torch", "cvxpy", "cvxpylayers", "diffcp", "clarabel")}}}
    for experiment, records in comparisons.items():
        pd.DataFrame(records).to_csv(out / f"{experiment}.csv", index=False, float_format="%.12g")
    daily.to_csv(out / "test_daily_returns.csv", float_format="%.15g")
    pd.concat(rebalance_frames, ignore_index=True).to_csv(out / "test_rebalance_weights.csv", index=False, float_format="%.12g")
    (out / "paired_bootstrap.json").write_text(json.dumps(bootstraps, indent=2), encoding="utf-8")
    (out / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _write_report(out, metadata, comparisons, bootstraps)
    print(f"Portfolio experiments saved to {out}")
    if any(row["Status"] != "ok" for records in comparisons.values() for row in records):
        raise RuntimeError(f"A requested portfolio run failed. See {out / 'experiment_metadata.json'}.")

import shutil
import subprocess

from kairos import config
from kairos.reporting.analysis import (
    discussion,
    findings,
    load_results,
    md_table,
    summary_points,
)

PCT = {c: "{:.2%}" for c in ("AnnReturn", "AnnVol", "MaxDrawdown", "CVaR5")}
SUMMARY_COLS = [
    "AnnReturn",
    "AnnVol",
    "Sharpe",
    "Sortino",
    "MaxDrawdown",
    "Calmar",
    "CVaR5",
    "AnnTurnover",
]


def fmt_prior():
    w = sorted(config.MARKET_WEIGHTS.items(), key=lambda kv: -kv[1])
    return ", ".join(f"{k} {v:.0%}" for k, v in w)


def build_markdown(paths=config.PATHS):
    r = load_results(paths)
    f = findings(r)
    fig = lambda name, cap: f"![{cap}](../results/figures/{name})\n\n*{cap}*\n"
    manifest = r["manifest"]
    models = r["models"].set_index("Asset")
    ic_cols = [c for c in models.columns if c.startswith("IC_")] + ["p_value_stacked"]
    skill_cols = [
        "BaseRate",
        "UpShare_stacked",
        "Hit_ensemble_stacked",
        "HitEdge_ensemble_stacked",
        "IC_ensemble_stacked",
        "R2OOS_ensemble_stacked",
        "R2OOS_ensemble_final",
        "R2OOS_xgb",
        "R2OOS_rf",
        "R2OOS_ridge",
        "R2OOS_lstm",
        "R2OOS_pooled_ridge",
        "R2OOS_pooled_xgb",
    ]
    sig = r["significance"][
        ["strategy", "benchmark", "diff", "ci_low", "ci_high", "p_value"]
    ]
    regime = r["regime_metrics"][
        ["AnnReturn", "Sharpe", "MaxDrawdown", "CVaR5", "Months"]
    ].copy()
    regime.index = [f"{a} / {b}" for a, b in regime.index]
    overlay_rows = [
        n
        for n in (
            "BL_ML_Regime",
            "BL_ML_Regime_IC",
            config.RAW,
            config.NAIVE,
            "Regime_60_40",
            "Static_60_40",
            "Regime_Market",
            "Market_Reference",
        )
        if n in r["summary"].index
    ]
    sens = r["sensitivity"].copy()
    sens["Default"] = sens["IsDefault"].map({True: "yes", False: ""})
    sens = sens[
        [
            "Parameter",
            "Value",
            "Default",
            "Variant",
            "Sharpe",
            "MaxDrawdown",
            "AnnTurnover",
            "AvgCash",
        ]
    ]
    skill = r["skill_period"].pivot_table(
        index="Forecast", columns="Period", values=["MeanIC", "MeanR2OOS"], sort=False
    )
    skill.columns = [f"{a} ({b})" for a, b in skill.columns]
    period = r["period"][
        ["AnnReturn", "AnnVol", "Sharpe", "MaxDrawdown", "Months"]
    ].copy()
    period.index = [f"{a} / {b}" for a, b in period.index]
    holdout_table = (
        md_table(
            r["holdout_metrics"][["AnnReturn", "AnnVol", "Sharpe", "MaxDrawdown"]],
            PCT | {"Sharpe": "{:.2f}"},
        )
        if r["holdout_metrics"] is not None
        else "No realised months have been added since the freeze date."
    )
    parts = [
        "# Kairos: Regime-Aware Portfolio Intelligence",
        "Track: Machine Learning (Deep Learning) Investment Strategies, Topic 2: Portfolio Allocation.",
        "## 1. Research Question",
        "Does a machine learning and deep learning driven approach to portfolio allocation, augmented with market regime awareness, "
        "produce statistically and economically meaningful improvements in risk-adjusted returns over classical allocation methods, "
        "and under what market conditions does this advantage hold or break down?",
        "## 2. Summary of Findings",
        *summary_points(r, f),
        "## 3. Data",
        "All prices are daily adjusted closes downloaded from Yahoo Finance through the yfinance library, with the CBOE VIX (^VIX) "
        "and the 13-week Treasury bill yield (^IRX) as the volatility and risk-free inputs. No series is simulated, interpolated "
        "or reconstructed. The panel is validated before use: history length, gaps, extreme daily moves, and sanity floors on "
        "cross-asset correlations (for example SPY and EFA) that would expose substitute or corrupted data. The manifest below "
        "records the row count, date range and a file hash for every cached series, so a fresh download can be checked against "
        "this run.",
        md_table(manifest, {"rows": "{:.0f}"}, index=False),
        "The investable universe expands over time. An asset enters the portfolio only after it has a valid walk-forward forecast and "
        f"at least {config.MIN_HISTORY} months of return history, so crypto assets enter when they have a usable track record.",
        "## 4. Method",
        "**Features.** About 20 causal technical, volatility, momentum, drawdown and market-beta features per asset, computed from daily "
        "prices, plus VIX level, change and z-score. Raw price levels are never used as features. Features are sampled at month end.",
        "**Regimes.** A two-state Gaussian HMM on SPY return, 21-day realised volatility and VIX. It is refit once a year on past data "
        "only, and daily states come from the forward filter, so the label on any day uses no later information. A volatility-percentile "
        "rule is the comparison method.",
        "**Forecasting.** One-month-ahead log return per asset. Candidates are walk-forward XGBoost, Random Forest and Ridge per asset, "
        "an LSTM written in NumPy whose training target is scaled by its training mean and standard deviation, two pooled cross-asset "
        "models (ridge and XGBoost, trained on all assets' deviations from their own expanding mean), and the expanding historical mean. "
        f"The original design stacked the first four models with non-negative weights fit on the previous {config.STACK_WINDOW} months "
        f"(ensemble_stacked). The final forecast stacks all candidates with convex weights (non-negative, summing to one) and then "
        f"shrinks the result toward the historical mean by the slope of realised on predicted deviations over the previous "
        f"{config.CALIBRATION_WINDOW} months, clipped to the range zero to one and set to zero until {config.CALIBRATION_MIN} months "
        f"exist. The first {config.INITIAL_TRAIN_MONTHS} months are used only for initial training. Skill is judged by the "
        "information coefficient, the hit rate against the always-up baseline, and the out-of-sample R2 against the expanding "
        "historical mean.",
        "**Forecast diagnostics.** The tree and linear models are re-fitted without the regime features, without all market-wide "
        f"stress inputs (regime and VIX), with a {config.DIAG_INITIAL_TRAIN}-month initial window, and with the first and last changes "
        "combined, and compared on the dates common to all designs. The link between the stress probability and next-month returns "
        "is measured separately inside and after the initial window. The LSTM is also re-run without target scaling to measure the "
        "effect of the repair.",
        "**Portfolio construction.** Black-Litterman with a Ledoit-Wolf covariance, a prior portfolio, and the calibrated forecasts as "
        "absolute views. View uncertainty follows the trailing forecast error variance. In the stressed regime, view uncertainty is "
        "multiplied and risk aversion is raised. The optimiser allows cash, so higher risk aversion lowers exposure. "
        "BL_ML_Regime_IC additionally scales each view by the asset's trailing IC. BL_ML_NoRegime removes the regime scaling and "
        "serves as the ablation. BL_ML_Raw_Regime uses the original uncalibrated stacked forecasts, and BL_HistMean_Regime uses the "
        "expanding historical mean as the view.",
        "**Priors.** The default prior is a hand-picked reference allocation. Alternatives are an equal-weight prior and a trailing "
        "inverse-volatility prior, which need no judgement, and an observed prior read from `data/raw/prior_weights.csv` when that file "
        "exists. Each Black-Litterman portfolio is compared with the portfolio that holds its own prior.",
        "**Benchmarks.** Market reference weights, equal weight, 60/40 (SPY and IEF), inverse volatility, and mean-variance with Ledoit-Wolf "
        "covariance. All benchmarks use the same eligible universe as the Black-Litterman portfolios. Regime_60_40 and Regime_Market "
        "apply the stress rule mechanically to 60/40 and to the market reference, with no forecasts.",
        "**Backtest.** Monthly rebalancing, 10 basis points on traded notional, weight drift, cash earning the T-bill rate, and Sharpe ratios "
        "measured on excess returns over that rate. Significance uses a paired stationary bootstrap of the Sharpe difference, centred on the "
        "null. A deflated Sharpe ratio allows for the Kairos variants tried. Sensitivity runs change one Black-Litterman parameter at a time.",
        "**Evaluation windows.** Results are reported for the full sample, for the periods before and from "
        f"{config.HOLDOUT_START}, and for a crisis-inclusive window with a {config.CRISIS_INITIAL_TRAIN}-month initial training period that "
        "starts the out-of-sample forecasts before 2008. A specification hash and freeze date are stored so that months realised after "
        "the freeze can be evaluated without any re-tuning.",
        "## 5. Results",
        "### 5.1 Regime detection",
        md_table(
            r["regime_validation"].set_index("Period"),
            {
                "Rate_hmm": "{:.1%}",
                "Rate_vol": "{:.1%}",
                "Days_hmm": "{:.0f}",
                "Days_vol": "{:.0f}",
                "FlaggedStressed_hmm": "{:.0f}",
                "FlaggedStressed_vol": "{:.0f}",
            },
        ),
        fig("03_regime_validation.png", "Share of days flagged stressed"),
        fig("02_regime_timeline.png", "Walk-forward regime timeline"),
        "### 5.2 Forecasting skill",
        "Out-of-sample information coefficient by asset and model:",
        md_table(models[ic_cols]),
        fig("04_model_ic.png", "Out-of-sample IC by asset and model"),
        "Hit rate against the always-up baseline and R2 against the historical mean (HitEdge is the hit rate minus the share of up months):",
        md_table(models[skill_cols]),
        fig("11_forecast_skill.png", "Forecast skill by asset"),
        fig(
            "05_shap_regime.png",
            "Regime-conditional SHAP importance (in-sample, for explanation only)",
        ),
        "### 5.3 Diagnosing the information coefficient",
        "Rank correlation between the filtered stress probability and the next-month return, inside and after the initial training window:",
        md_table(
            r["regime_link"].set_index("Asset"),
            {"n": "{:.0f}", "StressedMonths": "{:.0f}"},
        ),
        "Tree and linear models re-fitted under alternative designs, compared on common dates:",
        md_table(r["ic_summary"], {"Assets": "{:.0f}", "AssetsPositiveIC": "{:.0f}"}),
        fig("12_ic_diagnostics.png", "Mean IC by forecasting design"),
        "### 5.4 Repairing the forecasts",
        "LSTM before and after scaling the training target (Disp is the standard deviation of forecasts divided by that of realised returns):",
        md_table(r["lstm_repair"].set_index("Asset"), {"n_oos": "{:.0f}"}),
        "Mean information coefficient and mean R2 against the historical mean, by forecast and period:",
        md_table(skill),
        fig(
            "15_forecast_repair.png",
            "R2 against the historical mean by forecast and period",
        ),
        "Average stacking weight of each candidate in the final forecast:",
        md_table(r["stack_weights"], {c: "{:.2f}" for c in r["stack_weights"].columns}),
        "### 5.5 Portfolio performance",
        md_table(r["summary"][SUMMARY_COLS], {**PCT, "AnnTurnover": "{:.2f}"}),
        fig("07_cumulative_returns.png", "Cumulative net return"),
        fig("08_drawdowns.png", "Drawdowns"),
        fig("06_weights.png", "Portfolio weights"),
        "### 5.6 Statistical significance",
        md_table(sig, index=False),
        fig(
            "09_sharpe_significance.png",
            "Sharpe difference with 95% bootstrap interval",
        ),
        "Deflated Sharpe ratio of the Kairos variants (DSR is the probability that the true Sharpe ratio is positive after allowing for the number of trials):",
        md_table(r["deflated"], {"Trials": "{:.0f}"}, index=False),
        "### 5.7 Overlay ablation",
        md_table(
            r["summary"].loc[overlay_rows, SUMMARY_COLS],
            {**PCT, "AnnTurnover": "{:.2f}"},
        ),
        fig(
            "14_overlay_ablation.png",
            "Forecast-driven portfolios against mechanical overlays",
        ),
        "### 5.8 Performance by regime",
        md_table(regime, {**PCT, "Months": "{:.0f}"}),
        fig("10_regime_performance.png", "Annualised return by regime"),
        "### 5.9 Other evaluation windows",
        f"Performance before and from {config.HOLDOUT_START}:",
        md_table(period, {**PCT, "Months": "{:.0f}"}),
        "Paired Sharpe comparisons within each period:",
        md_table(
            r["period_sig"][
                [
                    "period",
                    "strategy",
                    "benchmark",
                    "diff",
                    "ci_low",
                    "ci_high",
                    "p_value",
                ]
            ],
            index=False,
        ),
        "Crisis-inclusive window (out-of-sample forecasts start before 2008):",
        md_table(
            r["window"],
            {
                **{c: "{:.3f}" for c in r["window"].columns},
                "Months_main": "{:.0f}",
                "Months_crisis_window": "{:.0f}",
            },
        ),
        md_table(
            r["window_sig"][
                ["strategy", "benchmark", "diff", "ci_low", "ci_high", "p_value"]
            ],
            index=False,
        ),
        fig("16_window_robustness.png", "Sharpe ratio by evaluation window"),
        "### 5.10 Robustness",
        "Sharpe ratio by transaction cost assumption:",
        md_table(r["cost"]),
        "Sharpe ratio by half of the sample:",
        md_table(r["subperiod"]),
        "Black-Litterman parameter sensitivity, one parameter changed at a time:",
        md_table(
            sens, {**PCT, "AnnTurnover": "{:.2f}", "AvgCash": "{:.1%}"}, index=False
        ),
        fig("13_sensitivity.png", "Sharpe ratio across parameter settings"),
        "Each prior against the portfolio that simply holds it:",
        md_table(
            r["prior"],
            {c: "{:.3f}" for c in ("Diff", "CI_low", "CI_high", "p_value")},
            index=False,
        ),
        fig(
            "17_prior_comparison.png",
            "Prior portfolios and the Black-Litterman portfolios built on them",
        ),
        "### 5.11 Frozen specification and new months",
        md_table(r["holdout_status"], index=False),
        holdout_table,
        "## 6. Discussion",
        *discussion(r, f),
        "## 7. Limitations",
        "Monthly data gives roughly 15 years of observations in the main window and about 20 in the crisis-inclusive window, which "
        "limits statistical power, so wide intervals should be read as inconclusive rather than as evidence of equivalence. "
        f"The default prior is a hand-picked reference allocation ({fmt_prior()}), not an observed market-capitalisation series. "
        "The inverse-volatility and equal-weight priors remove the judgement but are not market-implied either. "
        "The stress rule is calibrated by hand (multipliers fixed in advance) and the validation windows were chosen with hindsight, "
        "although they are not used in fitting. The forecast repairs and the diagnostics were designed after seeing full-sample "
        "results, so the development and holdout split is a reporting device and only months realised after the freeze date are "
        "untouched evidence. The LSTM is a small NumPy implementation refit every six months. Yahoo adjusted closes are "
        "retrospective vendor data and can be revised. The data stage needs network access to Yahoo Finance for a fresh download; "
        "later runs reuse the cache, and the manifest in section 3 allows the cached series to be checked. Results hold for this "
        "universe and period.",
        "## 8. Reproducibility",
        "Run `python scripts/run.py all` to download data, run every stage and rebuild this report, or `python scripts/run.py all --refresh` to "
        "download fresh prices first. Seeds, parameters and paths are in `code/kairos/config.py`.",
    ]
    return "\n\n".join(parts) + "\n"


def run(paths=config.PATHS):
    paths.reports.mkdir(parents=True, exist_ok=True)
    md = paths.reports / "Kairos_Report.md"
    md.write_text(build_markdown(paths), encoding="utf-8")
    if shutil.which("pandoc"):
        subprocess.run(
            [
                "pandoc",
                md.name,
                "-o",
                "Kairos_Report.docx",
                "--toc",
                "--resource-path=.",
            ],
            cwd=paths.reports,
            check=True,
        )
    return md

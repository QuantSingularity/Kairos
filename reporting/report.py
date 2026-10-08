import shutil
import subprocess

import config
from reporting.analysis import (
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
    q = r["quality"]
    prov = q[q["first_date"].notna()][["series", "first_date", "observations"]]
    models = r["models"].set_index("Asset")
    ic_cols = [c for c in models.columns if c.startswith("IC_")] + ["p_value_stacked"]
    skill_cols = [
        "BaseRate",
        "UpShare_stacked",
        "Hit_ensemble_stacked",
        "HitEdge_ensemble_stacked",
        "IC_ensemble_stacked",
        "R2OOS_ensemble_stacked",
        "R2OOS_xgb",
        "R2OOS_rf",
        "R2OOS_ridge",
        "R2OOS_lstm",
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
        "cross-asset correlations (for example SPY and EFA) that would expose substitute or corrupted data.",
        md_table(prov, {"observations": "{:.0f}"}, index=False),
        "The investable universe expands over time. An asset enters the portfolio only after it has a valid walk-forward forecast and "
        f"at least {config.MIN_HISTORY} months of return history, so crypto assets enter when they have a usable track record.",
        "## 4. Method",
        "**Features.** About 20 causal technical, volatility, momentum, drawdown and market-beta features per asset, computed from daily "
        "prices, plus VIX level, change and z-score. Raw price levels are never used as features. Features are sampled at month end.",
        "**Regimes.** A two-state Gaussian HMM on SPY return, 21-day realised volatility and VIX. It is refit once a year on past data "
        "only, and daily states come from the forward filter, so the label on any day uses no later information. A volatility-percentile "
        "rule is the comparison method.",
        "**Forecasting.** One-month-ahead log return per asset with walk-forward XGBoost, Random Forest, Ridge and a NumPy LSTM, plus an "
        f"equal-weight ensemble and a stacked ensemble whose non-negative weights are fit on the previous {config.STACK_WINDOW} months. "
        f"The first {config.INITIAL_TRAIN_MONTHS} months are used only for initial training. Skill is judged by the information "
        "coefficient, the hit rate against the always-up baseline, and the out-of-sample R2 against an expanding historical-mean forecast.",
        "**Forecast diagnostics.** The tree and linear models are re-fitted without the regime features, without all market-wide "
        f"stress inputs (regime and VIX), with a {config.DIAG_INITIAL_TRAIN}-month initial window, and with the first and last changes "
        "combined, and compared on the dates common to all designs. "
        "The link between the stress probability and next-month returns is measured separately inside and after the initial window.",
        "**Portfolio construction.** Black-Litterman with a Ledoit-Wolf covariance, a reference market-weight prior, and the stacked "
        "forecasts as absolute views. View uncertainty follows the trailing forecast error variance. In the stressed regime, view "
        "uncertainty is multiplied and risk aversion is raised. The optimiser allows cash, so higher risk aversion lowers exposure. "
        "BL_ML_Regime_IC additionally scales each view by the asset's trailing IC. BL_ML_NoRegime removes the regime scaling and "
        "serves as the ablation. BL_HistMean_Regime replaces the machine learning views with the expanding historical mean.",
        "**Benchmarks.** Market reference weights, equal weight, 60/40 (SPY and IEF), inverse volatility, and mean-variance with Ledoit-Wolf "
        "covariance. All benchmarks use the same eligible universe as the Black-Litterman portfolios. Regime_60_40 and Regime_Market "
        "apply the stress rule mechanically to 60/40 and to the market reference, with no forecasts.",
        "**Backtest.** Monthly rebalancing, 10 basis points on traded notional, weight drift, cash earning the T-bill rate, and Sharpe ratios "
        "measured on excess returns over that rate. Significance uses a paired stationary bootstrap of the Sharpe difference, centred on the "
        "null. A deflated Sharpe ratio allows for the Kairos variants tried. Sensitivity runs change one Black-Litterman parameter at a time.",
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
        fig("11_forecast_skill.png", "Forecast skill of the stacked ensemble"),
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
        "### 5.4 Portfolio performance",
        md_table(r["summary"][SUMMARY_COLS], {**PCT, "AnnTurnover": "{:.2f}"}),
        fig("07_cumulative_returns.png", "Cumulative net return"),
        fig("08_drawdowns.png", "Drawdowns"),
        fig("06_weights.png", "Portfolio weights"),
        "### 5.5 Statistical significance",
        md_table(sig, index=False),
        fig(
            "09_sharpe_significance.png",
            "Sharpe difference with 95% bootstrap interval",
        ),
        "Deflated Sharpe ratio of the Kairos variants (DSR is the probability that the true Sharpe ratio is positive after allowing for the number of trials):",
        md_table(r["deflated"], {"Trials": "{:.0f}"}, index=False),
        "### 5.6 Overlay ablation",
        md_table(
            r["summary"].loc[overlay_rows, SUMMARY_COLS],
            {**PCT, "AnnTurnover": "{:.2f}"},
        ),
        fig(
            "14_overlay_ablation.png",
            "Forecast-driven portfolios against mechanical overlays",
        ),
        "### 5.7 Performance by regime",
        md_table(regime, {**PCT, "Months": "{:.0f}"}),
        fig("10_regime_performance.png", "Annualised return by regime"),
        "### 5.8 Robustness",
        "Sharpe ratio by transaction cost assumption:",
        md_table(r["cost"]),
        "Sharpe ratio by half of the sample:",
        md_table(r["subperiod"]),
        "Black-Litterman parameter sensitivity, one parameter changed at a time:",
        md_table(
            sens, {**PCT, "AnnTurnover": "{:.2f}", "AvgCash": "{:.1%}"}, index=False
        ),
        fig("13_sensitivity.png", "Sharpe ratio across parameter settings"),
        "## 6. Discussion",
        *discussion(r, f),
        "## 7. Limitations",
        "The usable out-of-sample window begins after the initial training period, so the 2008 crisis is not part of the portfolio backtest "
        "even though its regime labels are available once the HMM has enough history. Monthly data gives roughly 15 years of "
        "observations, which limits statistical power, so wide intervals should be read as inconclusive rather than as evidence of equivalence. "
        f"The prior is a hand-picked reference allocation ({fmt_prior()}), not an observed market-capitalisation series, and the "
        "equal-weight prior check in section 5.8 shows how far the results depend on it. The stress rule is calibrated by hand "
        "(multipliers fixed in advance) and the validation windows were chosen with hindsight, although they are not used in fitting. "
        "The LSTM is a small NumPy implementation refit every six months. The diagnostics are exploratory: they use the same sample as the "
        "main test. Yahoo adjusted closes are retrospective vendor data and can be revised. Results hold for this universe and period.",
        "## 8. Reproducibility",
        "Run `python run.py all` to download data, run every stage and rebuild this report. Seeds, parameters and paths are in `config.py`.",
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

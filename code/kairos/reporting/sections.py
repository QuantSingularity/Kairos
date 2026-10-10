"""Results sections of the report."""

from typing import Any

from kairos import config
from kairos.reporting.narrative import discussion, summary_points
from kairos.reporting.results import md_table
from kairos.reporting.text import data_intro, universe_note

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


def fig(name: str, cap: str) -> str:
    """Markdown for a figure with its caption."""
    return f"![{cap}](../results/figures/{name})\n\n*{cap}*\n"


def intro_section(r: dict[str, Any], f: dict[str, Any]) -> list[str]:
    """Title, question and summary."""
    return [
        "# Kairos: Regime-Aware Portfolio Intelligence",
        "Track: Machine Learning (Deep Learning) Investment Strategies, Topic 2: Portfolio Allocation.",
        "## 1. Research Question",
        "Does a machine learning and deep learning driven approach to portfolio allocation, augmented with market regime awareness, "
        "produce statistically and economically meaningful improvements in risk-adjusted returns over classical allocation methods, "
        "and under what market conditions does this advantage hold or break down?",
        "## 2. Summary of Findings",
        *summary_points(r, f),
    ]


def data_section(r: dict[str, Any]) -> list[str]:
    """Data description and manifest."""
    return [
        "## 3. Data",
        *data_intro(),
        md_table(r["manifest"], {"rows": "{:.0f}"}, index=False),
        *universe_note(),
    ]


def forecasting_section(r: dict[str, Any]) -> list[str]:
    """Regime detection, forecasting and diagnostics."""
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
    skill = r["skill_period"].pivot_table(
        index="Forecast", columns="Period", values=["MeanIC", "MeanR2OOS"], sort=False
    )
    skill.columns = [f"{a} ({b})" for a, b in skill.columns]
    return [
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
    ]


def portfolio_section(r: dict[str, Any]) -> list[str]:
    """Performance, significance, overlays and regimes."""
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
    return [
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
    ]


def robustness_section(r: dict[str, Any]) -> list[str]:
    """Windows, sensitivity, priors and the frozen record."""
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
    return [
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
    ]


def discussion_section(r: dict[str, Any], f: dict[str, Any]) -> list[str]:
    """Discussion heading and paragraphs."""
    return [
        "## 6. Discussion",
        *discussion(r, f),
    ]

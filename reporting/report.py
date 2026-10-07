import shutil
import subprocess


import config
from reporting.analysis import findings, load_results, md_table

PCT = {c: "{:.2%}" for c in ("AnnReturn", "AnnVol", "MaxDrawdown", "CVaR5")}


def build_markdown(paths=config.PATHS):
    r = load_results(paths)
    f = findings(r)
    fig = lambda name, cap: f"![{cap}](../results/figures/{name})\n\n*{cap}*\n"
    q = r["quality"]
    prov = q[q["first_date"].notna()][["series", "first_date", "observations"]]
    models = r["models"].set_index("Asset")
    ic_cols = [c for c in models.columns if c.startswith("IC_")] + ["p_value_stacked"]
    summary = r["summary"][
        [
            "AnnReturn",
            "AnnVol",
            "Sharpe",
            "Sortino",
            "MaxDrawdown",
            "Calmar",
            "CVaR5",
            "AnnTurnover",
        ]
    ]
    sig = r["significance"][
        ["strategy", "benchmark", "diff", "ci_low", "ci_high", "p_value"]
    ]
    regime = r["regime_metrics"][
        ["AnnReturn", "Sharpe", "MaxDrawdown", "CVaR5", "Months"]
    ].copy()
    regime.index = [f"{a} / {b}" for a, b in regime.index]
    parts = [
        "# Kairos: Regime-Aware Portfolio Intelligence",
        "Track: Machine Learning (Deep Learning) Investment Strategies, Topic 2: Portfolio Allocation.",
        "## 1. Research Question",
        "Does a machine learning and deep learning driven approach to portfolio allocation, augmented with market regime awareness, "
        "produce statistically and economically meaningful improvements in risk-adjusted returns over classical allocation methods, "
        "and under what market conditions does this advantage hold or break down?",
        "## 2. Summary of Findings",
        f"{f['answer']}",
        f"Regime ablation: BL_ML_Regime versus BL_ML_NoRegime is {f['ablation']}. {f['stress_cvar']}",
        f"Forecasting: the stacked ensemble has a mean out-of-sample information coefficient of {f['mean_ic']:.3f}, and "
        f"{f['sig_assets']} of {f['n_assets']} assets show an IC significant at the 5% level. The stacked ensemble beats the mean of the "
        f"single models on {f['ens_beats']} of {f['n_assets']} assets.",
        f"Regimes: the walk-forward HMM flags {f['hmm_stress']:.0%} of days inside the documented stress windows and {f['hmm_calm']:.0%} "
        f"of days inside the calm control windows, against {f['vol_stress']:.0%} and {f['vol_calm']:.0%} for the volatility rule.",
        "Sharpe ranking: " + f["ranking"] + ".",
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
        f"The first {config.INITIAL_TRAIN_MONTHS} months are used only for initial training.",
        "**Portfolio construction.** Black-Litterman with a Ledoit-Wolf covariance, a reference market-weight prior, and the stacked "
        "forecasts as absolute views. View uncertainty follows the trailing forecast error variance. In the stressed regime, view "
        "uncertainty is multiplied and risk aversion is raised. The optimiser allows cash, so higher risk aversion lowers exposure. "
        "BL_ML_Regime_IC additionally scales each view by the asset's trailing IC. BL_ML_NoRegime removes the regime scaling and serves as the ablation.",
        "**Benchmarks.** Market reference weights, equal weight, 60/40 (SPY and IEF), inverse volatility, and mean-variance with Ledoit-Wolf "
        "covariance. All benchmarks use the same eligible universe as the Black-Litterman portfolios.",
        "**Backtest.** Monthly rebalancing, 10 basis points on traded notional, weight drift, cash earning the T-bill rate, and Sharpe ratios "
        "measured on excess returns over that rate. Significance uses a paired stationary bootstrap of the Sharpe difference, centred on the null.",
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
        "### 5.2 Forecasting",
        md_table(models[ic_cols]),
        fig("04_model_ic.png", "Out-of-sample IC by asset and model"),
        fig("05_shap_regime.png", "Regime-conditional SHAP importance"),
        "### 5.3 Portfolio performance",
        md_table(summary, {**PCT, "AnnTurnover": "{:.2f}"}),
        fig("07_cumulative_returns.png", "Cumulative net return"),
        fig("08_drawdowns.png", "Drawdowns"),
        fig("06_weights.png", "Portfolio weights"),
        "### 5.4 Statistical significance",
        md_table(sig, index=False),
        fig(
            "09_sharpe_significance.png",
            "Sharpe difference with 95% bootstrap interval",
        ),
        "### 5.5 Performance by regime",
        md_table(regime, {**PCT, "Months": "{:.0f}"}),
        fig("10_regime_performance.png", "Annualised return by regime"),
        "### 5.6 Robustness",
        "Sharpe ratio by transaction cost assumption:",
        md_table(r["cost"]),
        "Sharpe ratio by half of the sample:",
        md_table(r["subperiod"]),
        "## 6. Discussion",
        f"On the research question: {f['answer']} The regime ablation shows that regime scaling is {f['ablation']}.",
        "Where the advantage holds or breaks down is read from section 5.5: the regime table separates calm and stressed decision months, "
        "and the cash allowance means the regime-aware variants can reduce exposure in stress, which the ablation isolates.",
        "## 7. Limitations",
        "The usable out-of-sample window begins after the initial training period, so the 2008 crisis is not part of the portfolio backtest "
        "even though its regime labels are available once the HMM has enough history. Monthly data gives roughly 15 years of "
        "observations, which limits statistical power, so wide intervals should be read as inconclusive rather than as evidence of equivalence. "
        "The prior uses approximate global asset-class weights, not an observed market-capitalisation series. The LSTM is a small NumPy "
        "implementation refit every six months. Yahoo adjusted closes are retrospective vendor data and can be revised. Results hold for this universe and period.",
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

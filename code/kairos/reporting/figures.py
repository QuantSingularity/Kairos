"""Figures for the report and deck."""

import logging
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kairos import config
from kairos.data.store import load_panel, load_regimes, load_returns

log = logging.getLogger(__name__)

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "legend.frameon": False,
    }
)
COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#8c564b",
    "#7f7f7f",
    "#bcbd22",
    "#e377c2",
    "#393b79",
]
ORDER = [
    "BL_ML_Regime",
    "BL_ML_Regime_IC",
    "BL_ML_NoRegime",
    "BL_ML_Raw_Regime",
    "BL_HistMean_Regime",
    "Market_Reference",
    "Equal_Weight",
    "Static_60_40",
    "Inverse_Vol",
    "MeanVariance_LW",
    "Regime_60_40",
    "Regime_Market",
]
FOCUS = [
    "BL_ML_Regime",
    "BL_ML_Regime_IC",
    "BL_ML_NoRegime",
    "Market_Reference",
    "Static_60_40",
    "Regime_60_40",
    "Equal_Weight",
    "Inverse_Vol",
]


def _save(fig: Any, paths: config.Paths, name: str) -> None:
    """Save and close a figure."""
    fig.tight_layout()
    fig.savefig(paths.figures / name)
    plt.close(fig)


def asset_universe(panel: pd.DataFrame, paths: config.Paths) -> None:
    """Data availability by asset."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for a in config.ASSETS:
        if a in panel and a not in config.CRYPTO:
            s = panel[a].dropna()
            ax.plot(s.index, s / s.iloc[0], label=a, lw=1)
    ax.set_yscale("log")
    ax.set_title("Adjusted close, rebased to 1 at first observation (log scale)")
    ax.legend(ncol=4)
    _save(fig, paths, "01_asset_universe.png")


def regime_timeline(
    panel: pd.DataFrame, regimes: pd.DataFrame, paths: config.Paths
) -> None:
    """SPY with walk-forward stress periods."""
    fig, (a1, a2) = plt.subplots(
        2, 1, figsize=(8, 5), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    spy = panel["SPY"].dropna()
    a1.plot(spy.index, spy, color="#1f77b4", lw=1)
    a1.set_yscale("log")
    flag = regimes["regime"].dropna() == 1
    a1.fill_between(
        flag.index,
        spy.min(),
        spy.max(),
        where=flag.reindex(flag.index).values,
        color="#d62728",
        alpha=0.18,
        lw=0,
    )
    a1.set_title("SPY with walk-forward HMM stressed regime shaded")
    a2.plot(regimes.index, regimes["p_stressed"], color="#d62728", lw=0.8)
    a2.set_title("Filtered probability of the stressed state")
    _save(fig, paths, "02_regime_timeline.png")


def regime_validation(table: pd.DataFrame, paths: config.Paths) -> None:
    """Share of days flagged stressed by method."""
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(table))
    ax.bar(x - 0.2, table["Rate_hmm"], 0.4, label="HMM", color="#1f77b4")
    ax.bar(x + 0.2, table["Rate_vol"], 0.4, label="Volatility rule", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(table["Period"], rotation=30, ha="right")
    ax.set_ylabel("Share of days flagged stressed")
    ax.set_title(
        "Regime detection on known stress periods (left) and calm controls (right)"
    )
    ax.legend()
    _save(fig, paths, "03_regime_validation.png")


def model_ic(table: pd.DataFrame, paths: config.Paths) -> None:
    """Information coefficient by asset and model."""
    cols = [f"IC_{m}" for m in config.MODELS + config.POOLED] + ["IC_ensemble_stacked"]
    t = table.set_index("Asset")[[c for c in cols if c in table]]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    t.plot.bar(ax=ax, width=0.8, color=COLORS[: len(t.columns)])
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("Out-of-sample Spearman IC")
    ax.set_title("Forecast information coefficient by asset and model")
    ax.legend([c.replace("IC_", "") for c in t.columns], ncol=4, fontsize=6)
    plt.setp(ax.get_xticklabels(), rotation=0)
    _save(fig, paths, "04_model_ic.png")


def shap_regime(shap_tables: dict[str, pd.DataFrame], paths: config.Paths) -> None:
    """Regime-conditional SHAP importance."""
    if not shap_tables:
        return
    asset = "SPY" if "SPY" in shap_tables else next(iter(shap_tables))
    t = shap_tables[asset].head(10)[["calm", "stressed"]].iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    t.plot.barh(ax=ax, color=["#2ca02c", "#d62728"])
    ax.set_title(f"Mean absolute SHAP by regime, {asset} (top 10 features)")
    _save(fig, paths, "05_shap_regime.png")


def weights_chart(weights: dict[str, pd.DataFrame], paths: config.Paths) -> None:
    """Portfolio weights over time."""
    w = weights["BL_ML_Regime"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.stackplot(
        w.index,
        w.T.values,
        labels=w.columns,
        colors=plt.get_cmap("tab20")(np.linspace(0, 1, w.shape[1])),
    )
    ax.set_ylim(0, 1)
    ax.set_title("BL_ML_Regime portfolio weights (cash shown as CASH)")
    ax.legend(ncol=5, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    _save(fig, paths, "06_weights.png")


def cumulative_returns(returns: dict[str, pd.DataFrame], paths: config.Paths) -> None:
    """Cumulative net returns."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, n in enumerate(FOCUS):
        if n in returns:
            ax.plot(
                (1 + returns[n]["net"]).cumprod(),
                label=n,
                lw=1.6 if n.startswith("BL_") else 1,
                color=COLORS[i % len(COLORS)],
            )
    ax.set_yscale("log")
    ax.set_title("Cumulative net return (log scale)")
    ax.legend(ncol=2, fontsize=7)
    _save(fig, paths, "07_cumulative_returns.png")


def drawdowns(returns: dict[str, pd.DataFrame], paths: config.Paths) -> None:
    """Drawdown paths."""
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, n in enumerate(FOCUS):
        if n in returns:
            c = (1 + returns[n]["net"]).cumprod()
            ax.plot(c / c.cummax() - 1, label=n, lw=1, color=COLORS[i % len(COLORS)])
    ax.set_title("Drawdowns")
    ax.legend(ncol=2, fontsize=7)
    _save(fig, paths, "08_drawdowns.png")


def sharpe_significance(sig: pd.DataFrame, paths: config.Paths) -> None:
    """Sharpe differences with bootstrap intervals."""
    s = sig[sig["strategy"] == "BL_ML_Regime"].sort_values("diff")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    y = np.arange(len(s))
    ax.errorbar(
        s["diff"],
        y,
        xerr=[s["diff"] - s["ci_low"], s["ci_high"] - s["diff"]],
        fmt="o",
        color="#1f77b4",
        capsize=3,
    )
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(s["benchmark"])
    ax.set_title("BL_ML_Regime minus benchmark Sharpe, 95% bootstrap interval")
    _save(fig, paths, "09_sharpe_significance.png")


def regime_performance(regime_table: pd.DataFrame, paths: config.Paths) -> None:
    """Annualised return by regime."""
    t = regime_table.xs("Stressed", level=1)["AnnReturn"].reindex(ORDER).dropna()
    c = regime_table.xs("Calm", level=1)["AnnReturn"].reindex(t.index)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(t))
    ax.bar(x - 0.2, c.values, 0.4, label="Calm months", color="#2ca02c")
    ax.bar(x + 0.2, t.values, 0.4, label="Stressed months", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(t.index, rotation=30, ha="right")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title("Annualised return by regime at decision date")
    ax.legend()
    _save(fig, paths, "10_regime_performance.png")


def forecast_skill(table: pd.DataFrame, paths: config.Paths) -> None:
    """Forecast skill by asset."""
    t = table.dropna(subset=["IC_ensemble_stacked"]).set_index("Asset")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    t["IC_ensemble_stacked"].plot.bar(ax=axes[0], color="#1f77b4")
    axes[0].set_title("IC, original stacked ensemble")
    x = np.arange(len(t))
    axes[1].bar(
        x - 0.2,
        t["R2OOS_ensemble_stacked"],
        0.4,
        label="Original stacked",
        color="#d62728",
    )
    axes[1].bar(
        x + 0.2, t["R2OOS_ensemble_final"], 0.4, label="Calibrated", color="#2ca02c"
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(t.index)
    axes[1].set_title("R2 against historical mean")
    axes[1].legend(fontsize=7)
    axes[2].bar(
        x - 0.2, t["BaseRate"], 0.4, label="Always-up baseline", color="#7f7f7f"
    )
    axes[2].bar(
        x + 0.2,
        t["Hit_ensemble_stacked"],
        0.4,
        label="Original stacked hit rate",
        color="#2ca02c",
    )
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(t.index)
    axes[2].set_ylim(0.3, 0.75)
    axes[2].set_title("Hit rate against baseline")
    axes[2].legend(fontsize=7)
    for ax in axes:
        ax.axhline(0, color="black", lw=0.6)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    _save(fig, paths, "11_forecast_skill.png")


def ic_diagnostics(
    summary: pd.DataFrame, table: pd.DataFrame, paths: config.Paths
) -> None:
    """Mean IC by forecasting design."""
    names = list(summary.index)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    x = np.arange(len(names))
    axes[0].bar(x - 0.2, summary["MeanIC"], 0.4, label="All assets", color="#1f77b4")
    axes[0].bar(
        x + 0.2, summary["MeanIC_equities"], 0.4, label="SPY and EFA", color="#ff7f0e"
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    axes[0].axhline(0, color="black", lw=0.6)
    axes[0].set_title("Mean IC by forecasting design")
    axes[0].legend(fontsize=7)
    axes[1].bar(x, summary["MeanR2OOS"], 0.5, color="#d62728")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    axes[1].axhline(0, color="black", lw=0.6)
    axes[1].set_title("Mean R2 against historical mean")
    _save(fig, paths, "12_ic_diagnostics.png")


def sensitivity_chart(table: pd.DataFrame, paths: config.Paths) -> None:
    """Sharpe across parameter settings."""
    params = list(table["Parameter"].unique())
    cols = 3
    rows = int(np.ceil(len(params) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.8 * rows), squeeze=False)
    ref = float(table["Sharpe_60_40"].iloc[0])
    for ax, name in zip(axes.ravel(), params):
        sub = table[table["Parameter"] == name]
        for i, v in enumerate(sub["Variant"].unique()):
            part = sub[sub["Variant"] == v]
            ax.plot(part["Value"], part["Sharpe"], marker="o", label=v, color=COLORS[i])
        ax.axhline(ref, color="black", ls="--", lw=0.8, label="Static_60_40")
        ax.set_title(name)
    for ax in axes.ravel()[len(params) :]:
        ax.axis("off")
    axes[0][0].legend(fontsize=6)
    _save(fig, paths, "13_sensitivity.png")


def overlay_ablation(summary: pd.DataFrame, paths: config.Paths) -> None:
    """Forecast-driven portfolios against mechanical overlays."""
    names = [
        n
        for n in (
            "BL_ML_Regime",
            "BL_ML_Regime_IC",
            "BL_ML_Raw_Regime",
            "BL_HistMean_Regime",
            "Regime_60_40",
            "Static_60_40",
            "Regime_Market",
            "Market_Reference",
        )
        if n in summary.index
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    colors = [COLORS[i] for i in range(len(names))]
    axes[0].barh(
        names[::-1], summary.loc[names, "Sharpe"].values[::-1], color=colors[::-1]
    )
    axes[0].set_title("Sharpe ratio")
    axes[1].barh(
        names[::-1], summary.loc[names, "MaxDrawdown"].values[::-1], color=colors[::-1]
    )
    axes[1].set_title("Maximum drawdown")
    axes[1].set_yticklabels([])
    _save(fig, paths, "14_overlay_ablation.png")


def forecast_repair(skill: pd.DataFrame, paths: config.Paths) -> None:
    """R2 against the historical mean by forecast and period."""
    t = skill[skill["Period"].isin(["development", "holdout"])]
    wide = t.pivot(index="Forecast", columns="Period", values="MeanR2OOS")
    order = [f for f in skill["Forecast"].unique() if f in wide.index]
    wide = wide.reindex(order)[["development", "holdout"]]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = np.arange(len(wide))
    ax.bar(
        x - 0.2,
        wide["development"],
        0.4,
        label=f"Before {config.HOLDOUT_START[:4]}",
        color="#1f77b4",
    )
    ax.bar(
        x + 0.2,
        wide["holdout"],
        0.4,
        label=f"From {config.HOLDOUT_START[:4]}",
        color="#ff7f0e",
    )
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(wide.index, rotation=30, ha="right", fontsize=8)
    ax.set_title("Mean R2 against the historical mean, by forecast and period")
    ax.legend(fontsize=8)
    _save(fig, paths, "15_forecast_repair.png")


def window_robustness(table: pd.DataFrame, paths: config.Paths) -> None:
    """Sharpe ratio by evaluation window."""
    names = [n for n in ORDER if n in table.index]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = np.arange(len(names))
    ax.bar(
        x - 0.2,
        table.loc[names, "Sharpe_main"],
        0.4,
        label="Main window",
        color="#1f77b4",
    )
    ax.bar(
        x + 0.2,
        table.loc[names, "Sharpe_crisis_window"],
        0.4,
        label="Crisis-inclusive window",
        color="#d62728",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_title("Sharpe ratio by evaluation window")
    ax.legend(fontsize=8)
    _save(fig, paths, "16_window_robustness.png")


def prior_comparison(table: pd.DataFrame, paths: config.Paths) -> None:
    """Prior portfolios and the portfolios built on them."""
    priors = list(table["Prior"].unique())
    variants = list(table["Variant"].unique())
    fig, ax = plt.subplots(figsize=(9, 3.8))
    width = 0.8 / (len(variants) + 1)
    x = np.arange(len(priors))
    base = (
        table.drop_duplicates("Prior")
        .set_index("Prior")
        .loc[priors, "Sharpe_PriorPortfolio"]
    )
    ax.bar(x - 0.4 + width / 2, base, width, label="Prior portfolio", color="#7f7f7f")
    for i, v in enumerate(variants):
        sub = table[table["Variant"] == v].set_index("Prior").loc[priors, "Sharpe"]
        ax.bar(x - 0.4 + width * (i + 1.5), sub, width, label=v, color=COLORS[i])
    ax.set_xticks(x)
    ax.set_xticklabels(priors)
    ax.set_title(
        "Sharpe ratio of each prior portfolio and of the Black-Litterman portfolios built on it"
    )
    ax.title.set_fontsize(9)
    ax.legend(fontsize=7)
    _save(fig, paths, "17_prior_comparison.png")


def run(paths: config.Paths = config.PATHS) -> None:
    """Figures stage."""
    paths.figures.mkdir(parents=True, exist_ok=True)
    panel = load_panel(paths)
    regimes = load_regimes(paths)
    asset_universe(panel, paths)
    regime_timeline(panel, regimes, paths)
    regime_validation(pd.read_csv(paths.tables / "regime_validation.csv"), paths)
    model_ic(pd.read_csv(paths.tables / "model_comparison.csv"), paths)
    shap_tables = {}
    for a in config.SHAP_ASSETS:
        f = paths.tables / f"shap_{a.replace('-', '_')}.csv"
        if f.exists():
            shap_tables[a] = pd.read_csv(f, index_col=0)
    shap_regime(shap_tables, paths)
    ret = load_returns(paths)
    returns = {n: pd.DataFrame({"net": ret[n]}) for n in ret.columns}
    weights = {
        n: pd.read_csv(
            paths.processed / f"weights_{n}.csv", index_col=0, parse_dates=True
        )
        for n in ("BL_ML_Regime",)
    }
    weights_chart(weights, paths)
    cumulative_returns(returns, paths)
    drawdowns(returns, paths)
    sharpe_significance(pd.read_csv(paths.tables / "significance.csv"), paths)
    regime_performance(
        pd.read_csv(paths.tables / "regime_metrics.csv", index_col=[0, 1]), paths
    )
    forecast_skill(pd.read_csv(paths.tables / "model_comparison.csv"), paths)
    ic_diagnostics(
        pd.read_csv(paths.tables / "ic_diagnostics_summary.csv", index_col=0),
        pd.read_csv(paths.tables / "ic_diagnostics.csv"),
        paths,
    )
    sensitivity_chart(pd.read_csv(paths.tables / "sensitivity.csv"), paths)
    overlay_ablation(
        pd.read_csv(paths.tables / "summary_metrics.csv", index_col=0), paths
    )
    forecast_repair(pd.read_csv(paths.tables / "skill_by_period.csv"), paths)
    window_robustness(
        pd.read_csv(paths.tables / "window_robustness.csv", index_col=0), paths
    )
    prior_comparison(pd.read_csv(paths.tables / "prior_comparison.csv"), paths)
    log.info("figures written to %s", paths.figures)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

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
]
FOCUS = [
    "BL_ML_Regime",
    "BL_ML_Regime_IC",
    "BL_ML_NoRegime",
    "Market_Reference",
    "Equal_Weight",
    "Static_60_40",
    "Inverse_Vol",
    "MeanVariance_LW",
]


def _save(fig, paths, name):
    fig.tight_layout()
    fig.savefig(paths.figures / name)
    plt.close(fig)


def asset_universe(panel, paths):
    fig, ax = plt.subplots(figsize=(8, 4))
    for a in config.ASSETS:
        if a in panel and a not in config.CRYPTO:
            s = panel[a].dropna()
            ax.plot(s.index, s / s.iloc[0], label=a, lw=1)
    ax.set_yscale("log")
    ax.set_title("Adjusted close, rebased to 1 at first observation (log scale)")
    ax.legend(ncol=4)
    _save(fig, paths, "01_asset_universe.png")


def regime_timeline(panel, regimes, paths):
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


def regime_validation(table, paths):
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


def model_ic(table, paths):
    cols = [f"IC_{m}" for m in config.MODELS] + ["IC_ensemble_stacked"]
    t = table.set_index("Asset")[[c for c in cols if c in table]]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    t.plot.bar(ax=ax, width=0.8, color=COLORS[: len(t.columns)])
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("Out-of-sample Spearman IC")
    ax.set_title("Forecast information coefficient by asset and model")
    ax.legend([c.replace("IC_", "") for c in t.columns], ncol=5, fontsize=7)
    plt.setp(ax.get_xticklabels(), rotation=0)
    _save(fig, paths, "04_model_ic.png")


def shap_regime(shap_tables, paths):
    if not shap_tables:
        return
    asset = "SPY" if "SPY" in shap_tables else next(iter(shap_tables))
    t = shap_tables[asset].head(10)[["calm", "stressed"]].iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    t.plot.barh(ax=ax, color=["#2ca02c", "#d62728"])
    ax.set_title(f"Mean absolute SHAP by regime, {asset} (top 10 features)")
    _save(fig, paths, "05_shap_regime.png")


def weights_chart(weights, paths):
    w = weights["BL_ML_Regime"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.stackplot(
        w.index,
        w.T.values,
        labels=w.columns,
        colors=plt.cm.tab20(np.linspace(0, 1, w.shape[1])),
    )
    ax.set_ylim(0, 1)
    ax.set_title("BL_ML_Regime portfolio weights (cash shown as CASH)")
    ax.legend(ncol=5, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    _save(fig, paths, "06_weights.png")


def cumulative_returns(returns, paths):
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, n in enumerate(FOCUS):
        if n in returns:
            ax.plot(
                (1 + returns[n]["net"]).cumprod(),
                label=n,
                lw=1.6 if n.startswith("BL_") else 1,
                color=COLORS[i % 8],
            )
    ax.set_yscale("log")
    ax.set_title("Cumulative net return (log scale)")
    ax.legend(ncol=2, fontsize=7)
    _save(fig, paths, "07_cumulative_returns.png")


def drawdowns(returns, paths):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, n in enumerate(FOCUS):
        if n in returns:
            c = (1 + returns[n]["net"]).cumprod()
            ax.plot(c / c.cummax() - 1, label=n, lw=1, color=COLORS[i % 8])
    ax.set_title("Drawdowns")
    ax.legend(ncol=2, fontsize=7)
    _save(fig, paths, "08_drawdowns.png")


def sharpe_significance(sig, paths):
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


def regime_performance(regime_table, paths):
    t = regime_table.xs("Stressed", level=1)["AnnReturn"].reindex(FOCUS).dropna()
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


def run(paths=config.PATHS):
    paths.figures.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(
        paths.processed / "panel_daily.csv", index_col=0, parse_dates=True
    )
    regimes = pd.read_csv(
        paths.processed / "regimes_daily.csv", index_col=0, parse_dates=True
    )
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
    ret = pd.read_csv(
        paths.processed / "backtest_returns.csv", index_col=0, parse_dates=True
    )
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

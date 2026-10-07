import numpy as np
import pandas as pd

import config
from backtest import strategies as st
from backtest.stats import paired_sharpe_test
from data.loader import month_end_dates
from models.black_litterman import black_litterman_weights, build_universe
from models.forecasting import load_forecasts


def monthly_data(panel):
    dates = month_end_dates(panel.index)
    prices = panel[[a for a in config.ASSETS if a in panel]].reindex(dates)
    return prices, np.log(prices).diff(), panel["RF"].reindex(dates) / 12.0


def portfolio_returns(weights, simple_rets, rf, cost_bps=config.COST_BPS):
    idx, asset_cols = simple_rets.index, [c for c in weights.columns if c != "CASH"]
    rows, prev = [], None
    for d in weights.index:
        pos = idx.get_loc(d)
        if pos + 1 >= len(idx):
            break
        w = weights.loc[d].values
        r = np.append(
            simple_rets.loc[idx[pos + 1]].reindex(asset_cols).fillna(0.0).values,
            rf.at[d],
        )
        gross = float(w @ r)
        turnover = float(
            np.abs(w[:-1] - (prev[:-1] if prev is not None else 0.0)).sum()
        )
        cost = turnover * cost_bps / 10000.0
        rows.append((d, gross, turnover, cost, gross - cost, rf.at[d]))
        prev = w * (1.0 + r) / (1.0 + gross)
    return pd.DataFrame(
        rows, columns=["date", "gross", "turnover", "cost", "net", "rf"]
    ).set_index("date")


def performance(bt):
    net, excess = bt["net"], bt["net"] - bt["rf"]
    n = len(net)
    ann = (1 + net).prod() ** (12 / n) - 1
    vol = net.std() * np.sqrt(12)
    downside = np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2))
    curve = (1 + net).cumprod()
    max_dd = float((curve / curve.cummax() - 1).min())
    k = max(1, int(np.ceil(0.05 * n)))
    return {
        "AnnReturn": ann,
        "AnnVol": vol,
        "Sharpe": excess.mean() / excess.std() * np.sqrt(12),
        "Sortino": excess.mean() / downside * np.sqrt(12) if downside > 0 else np.nan,
        "MaxDrawdown": max_dd,
        "Calmar": ann / abs(max_dd) if max_dd < 0 else np.nan,
        "CVaR5": float(net.nsmallest(k).mean()),
        "AnnTurnover": float(bt["turnover"].mean() * 12),
        "Months": n,
    }


def regime_performance(bt, regime):
    r = regime.reindex(bt.index)
    rows = {"All": performance(bt)}
    for label, val in (("Calm", 0.0), ("Stressed", 1.0)):
        sub = bt[r == val]
        rows[label] = (
            performance(sub) if len(sub) >= 6 else {k: np.nan for k in rows["All"]}
        )
        rows[label]["Months"] = len(sub)
    return pd.DataFrame(rows).T


def build_strategies(forecasts, log_rets, rf, regime):
    universe = build_universe(forecasts, log_rets)
    excess = log_rets.sub(rf, axis=0)
    weights = {
        "BL_ML_Regime": black_litterman_weights(
            forecasts, log_rets, rf, regime, universe, True, False
        ),
        "BL_ML_Regime_IC": black_litterman_weights(
            forecasts, log_rets, rf, regime, universe, True, True
        ),
        "BL_ML_NoRegime": black_litterman_weights(
            forecasts, log_rets, rf, regime, universe, False, False
        ),
        "Market_Reference": st.market_reference(universe),
        "Equal_Weight": st.equal_weight(universe),
        "Static_60_40": st.static_60_40(universe),
        "Inverse_Vol": st.inverse_vol(universe, log_rets),
        "MeanVariance_LW": st.mean_variance(universe, excess),
    }
    return universe, weights


def run_backtest(forecasts, panel, regime, cost_grid=config.COST_GRID):
    _, log_rets, rf = monthly_data(panel)
    simple = np.expm1(log_rets)
    universe, weights = build_strategies(forecasts, log_rets, rf, regime)
    runs = {
        c: {n: portfolio_returns(w, simple, rf, c) for n, w in weights.items()}
        for c in cost_grid
    }
    main = runs[config.COST_BPS]
    summary = pd.DataFrame({n: performance(bt) for n, bt in main.items()}).T
    regime_table = pd.concat(
        {n: regime_performance(bt, regime) for n, bt in main.items()}
    )
    cost_table = pd.DataFrame(
        {
            c: {n: performance(bt)["Sharpe"] for n, bt in r.items()}
            for c, r in runs.items()
        }
    )
    cost_table.columns = [f"{c}bps" for c in cost_table.columns]
    half = len(next(iter(main.values()))) // 2
    sub = {}
    for n, bt in main.items():
        sub[n] = {
            "FirstHalfSharpe": performance(bt.iloc[:half])["Sharpe"],
            "SecondHalfSharpe": performance(bt.iloc[half:])["Sharpe"],
        }
    excess = {n: bt["net"] - bt["rf"] for n, bt in main.items()}
    rows = []
    for a in ("BL_ML_Regime", "BL_ML_Regime_IC"):
        for b in excess:
            if a != b:
                rows.append(
                    {
                        "strategy": a,
                        "benchmark": b,
                        **paired_sharpe_test(excess[a], excess[b]),
                    }
                )
    return {
        "weights": weights,
        "returns": main,
        "summary": summary,
        "regime": regime_table,
        "cost": cost_table,
        "subperiod": pd.DataFrame(sub).T,
        "significance": pd.DataFrame(rows),
    }


def run(paths=config.PATHS):
    panel = pd.read_csv(
        paths.processed / "panel_daily.csv", index_col=0, parse_dates=True
    )
    F = pd.read_csv(
        paths.processed / "features_monthly.csv", index_col=0, parse_dates=True
    )
    out = run_backtest(load_forecasts(paths), panel, F["regime"])
    out["summary"].to_csv(paths.tables / "summary_metrics.csv")
    out["regime"].to_csv(paths.tables / "regime_metrics.csv")
    out["cost"].to_csv(paths.tables / "cost_sensitivity.csv")
    out["subperiod"].to_csv(paths.tables / "subperiod_sharpe.csv")
    out["significance"].to_csv(paths.tables / "significance.csv", index=False)
    pd.DataFrame({n: bt["net"] for n, bt in out["returns"].items()}).to_csv(
        paths.processed / "backtest_returns.csv"
    )
    pd.DataFrame({n: bt["rf"] for n, bt in out["returns"].items()}).iloc[:, :1].rename(
        columns=lambda c: "rf"
    ).to_csv(paths.processed / "backtest_rf.csv")
    for name in ("BL_ML_Regime", "BL_ML_Regime_IC", "BL_ML_NoRegime"):
        out["weights"][name].to_csv(paths.processed / f"weights_{name}.csv")
    return out

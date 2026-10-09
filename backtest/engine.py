import numpy as np
import pandas as pd

import config
from backtest import strategies as st
from backtest.stats import deflated_sharpe, paired_sharpe_test
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


def naive_forecasts(forecasts):
    view = config.VIEW_COLUMN
    return {
        a: df.assign(**{view: df["hist_mean"].where(df[view].notna())})
        for a, df in forecasts.items()
    }


def build_strategies(forecasts, log_rets, rf, regime):
    universe = build_universe(forecasts, log_rets)
    excess = log_rets.sub(rf, axis=0)
    raw = config.RAW_VIEW_COLUMN
    naive = naive_forecasts(forecasts)
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
        config.RAW: black_litterman_weights(
            forecasts, log_rets, rf, regime, universe, True, False, view=raw
        ),
        config.NAIVE: black_litterman_weights(
            naive, log_rets, rf, regime, universe, True, False
        ),
        "Market_Reference": st.market_reference(universe),
        "Equal_Weight": st.equal_weight(universe),
        "Static_60_40": st.static_60_40(universe),
        "Inverse_Vol": st.inverse_vol(universe, log_rets),
        "MeanVariance_LW": st.mean_variance(universe, excess),
    }
    weights["Regime_60_40"] = st.regime_overlay(weights["Static_60_40"], regime)
    weights["Regime_Market"] = st.regime_overlay(weights["Market_Reference"], regime)
    return universe, weights


def significance_pairs(names):
    pairs = [
        (a, b) for a in ("BL_ML_Regime", "BL_ML_Regime_IC") for b in names if a != b
    ]
    pairs += [
        ("Regime_60_40", "Static_60_40"),
        ("Regime_Market", "Market_Reference"),
        ("Regime_60_40", "BL_ML_Regime"),
        ("Regime_Market", "BL_ML_Regime"),
        ("BL_ML_Regime_IC", "Regime_60_40"),
        ("BL_ML_Regime", config.RAW),
    ]
    return list(dict.fromkeys(pairs))


def deflated_table(excess):
    pool = [n for n in list(config.KAIROS) + list(config.ABLATIONS) if n in excess]
    monthly = {n: excess[n].mean() / excess[n].std(ddof=1) for n in pool}
    sr_var = float(np.var(list(monthly.values()), ddof=1)) if len(pool) > 1 else 0.0
    rows = []
    for n in [n for n in config.KAIROS if n in excess]:
        for trials in config.DSR_TRIALS:
            rows.append({"strategy": n, **deflated_sharpe(excess[n], trials, sr_var)})
    return pd.DataFrame(rows)


PERIOD_PAIRS = [
    ("BL_ML_Regime", "Static_60_40"),
    ("BL_ML_Regime", "Regime_60_40"),
    ("BL_ML_Regime", config.NAIVE),
    ("BL_ML_Regime", config.RAW),
    ("Regime_60_40", "Static_60_40"),
]


def period_tables(main, excess):
    cut = pd.Timestamp(config.HOLDOUT_START)
    parts = (("development", lambda i: i < cut), ("holdout", lambda i: i >= cut))
    rows = {}
    for n, bt in main.items():
        for label, pick in parts:
            sub = bt[pick(bt.index)]
            if len(sub) >= 12:
                m = performance(sub)
                rows[(n, label)] = {
                    k: m[k] for k in ("AnnReturn", "AnnVol", "Sharpe", "MaxDrawdown")
                } | {"Months": len(sub)}
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index = pd.MultiIndex.from_tuples(table.index, names=["strategy", "period"])
    tests = []
    for label, pick in parts:
        for a, b in PERIOD_PAIRS:
            if a in excess and b in excess:
                mask = pick(excess[a].index)
                if mask.sum() >= 12:
                    tests.append(
                        {
                            "period": label,
                            "strategy": a,
                            "benchmark": b,
                            **paired_sharpe_test(excess[a][mask], excess[b][mask]),
                        }
                    )
    return table, pd.DataFrame(tests)


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
    period = period_tables(main, excess)
    rows = [
        {"strategy": a, "benchmark": b, **paired_sharpe_test(excess[a], excess[b])}
        for a, b in significance_pairs(list(excess))
    ]
    return {
        "weights": weights,
        "returns": main,
        "summary": summary,
        "regime": regime_table,
        "cost": cost_table,
        "subperiod": pd.DataFrame(sub).T,
        "significance": pd.DataFrame(rows),
        "deflated": deflated_table(excess),
        "period": period[0],
        "period_significance": period[1],
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
    out["deflated"].to_csv(paths.tables / "deflated_sharpe.csv", index=False)
    out["period"].to_csv(paths.tables / "period_split.csv")
    out["period_significance"].to_csv(
        paths.tables / "period_significance.csv", index=False
    )
    pd.DataFrame({n: bt["net"] for n, bt in out["returns"].items()}).to_csv(
        paths.processed / "backtest_returns.csv"
    )
    pd.DataFrame({n: bt["rf"] for n, bt in out["returns"].items()}).iloc[:, :1].rename(
        columns=lambda c: "rf"
    ).to_csv(paths.processed / "backtest_rf.csv")
    for name in ("BL_ML_Regime", "BL_ML_Regime_IC", "BL_ML_NoRegime"):
        out["weights"][name].to_csv(paths.processed / f"weights_{name}.csv")
    return out

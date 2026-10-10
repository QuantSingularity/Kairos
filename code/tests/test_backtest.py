import numpy as np
import pandas as pd

from kairos import config
from kairos.backtest import engine, stats, strategies


def toy_returns(n=10, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-31", periods=n, freq="ME")
    return pd.DataFrame(
        rng.normal(0.01, 0.03, (n, len(config.ASSETS))),
        index=idx,
        columns=config.ASSETS,
    ), pd.Series(0.002, index=idx)


def weights_frame(index, spec):
    w = pd.DataFrame(0.0, index=index, columns=config.ASSETS + ["CASH"])
    for k, v in spec.items():
        w[k] = v
    return w


def test_timing_uses_next_month_return():
    rets, rf = toy_returns()
    w = weights_frame(rets.index[:5], {"SPY": 1.0})
    bt = engine.portfolio_returns(w, rets, rf, cost_bps=0)
    assert np.isclose(bt["gross"].iloc[0], rets["SPY"].iloc[1])
    assert np.isclose(bt["gross"].iloc[3], rets["SPY"].iloc[4])


def test_last_decision_date_without_next_return_is_dropped():
    rets, rf = toy_returns()
    w = weights_frame(rets.index, {"SPY": 1.0})
    assert len(engine.portfolio_returns(w, rets, rf, 0)) == len(rets) - 1


def test_cash_earns_risk_free_rate():
    rets, rf = toy_returns()
    w = weights_frame(rets.index[:4], {"CASH": 1.0})
    bt = engine.portfolio_returns(w, rets, rf, 0)
    assert np.allclose(bt["gross"], 0.002) and np.allclose(bt["turnover"], 0.0)


def test_turnover_includes_drift_and_initial_purchase():
    rets, rf = toy_returns()
    w = weights_frame(rets.index[:4], {"SPY": 0.5, "TLT": 0.5})
    bt = engine.portfolio_returns(w, rets, rf, cost_bps=10)
    assert np.isclose(bt["turnover"].iloc[0], 1.0)
    r1 = rets.iloc[1][["SPY", "TLT"]].values
    drifted = 0.5 * (1 + r1) / (1 + 0.5 * r1.sum())
    assert np.isclose(bt["turnover"].iloc[1], np.abs(0.5 - drifted).sum())
    assert np.isclose(bt["cost"].iloc[0], 1.0 * 10 / 1e4)


def test_performance_metrics_known_values():
    idx = pd.date_range("2020-01-31", periods=12, freq="ME")
    net = [0.10, -0.20, 0.05] + [0.0] * 9
    bt = pd.DataFrame({"net": net, "rf": 0.0, "turnover": 0.1}, index=idx)
    m = engine.performance(bt)
    assert np.isclose(m["AnnReturn"], 1.1 * 0.8 * 1.05 - 1)
    assert np.isclose(m["MaxDrawdown"], 0.88 / 1.1 - 1)
    assert np.isclose(m["CVaR5"], -0.20)
    assert np.isclose(m["AnnTurnover"], 1.2)
    assert (
        m["Sharpe"]
        < 0
        < engine.performance(bt.assign(net=0.01 + np.arange(12) * 1e-4))["Sharpe"]
    )


def test_costs_reduce_returns():
    rets, rf = toy_returns(20)
    w = weights_frame(rets.index[:15], {"SPY": 1.0})
    w.iloc[::2] = 0.0
    w.loc[w.index[::2], "TLT"] = 1.0
    cheap = engine.portfolio_returns(w, rets, rf, 0)["net"].sum()
    dear = engine.portfolio_returns(w, rets, rf, 50)["net"].sum()
    assert dear < cheap


def test_stationary_bootstrap_indices_valid():
    idx = stats.stationary_indices(50, 200, 6, np.random.default_rng(0))
    assert idx.min() >= 0 and idx.max() < 50
    steps = np.diff(idx, axis=1) % 50
    assert 0.7 < (steps == 1).mean() < 0.95


def test_sharpe_test_is_calibrated_under_null():
    rng = np.random.default_rng(11)
    pvals = []
    for k in range(60):
        base = rng.normal(0.005, 0.04, 120)
        a = pd.Series(base + rng.normal(0, 0.01, 120))
        b = pd.Series(base + rng.normal(0, 0.01, 120))
        pvals.append(stats.paired_sharpe_test(a, b, draws=400, seed=k)["p_value"])
    assert 0.02 <= np.mean(np.array(pvals) < 0.05) <= 0.25


def test_sharpe_test_detects_real_difference():
    rng = np.random.default_rng(5)
    base = rng.normal(0, 0.03, 180)
    a, b = pd.Series(base + 0.02), pd.Series(base)
    r = stats.paired_sharpe_test(a, b, draws=1000)
    assert r["diff"] > 0 and r["p_value"] < 0.01 and r["ci_low"] > 0


def test_identical_series_have_no_difference():
    s = pd.Series(np.random.default_rng(1).normal(0.01, 0.03, 100))
    r = stats.paired_sharpe_test(s, s.copy(), draws=300)
    assert r["diff"] == 0 and r["p_value"] > 0.9


def test_benchmark_weights_share_universe_and_sum_to_one():
    rets, rf = toy_returns(80)
    universe = {d: ["SPY", "IEF", "TLT", "GLD"] for d in rets.index[40:]}
    excess = rets.sub(rf, axis=0)
    frames = [
        strategies.equal_weight(universe),
        strategies.market_reference(universe),
        strategies.static_60_40(universe),
        strategies.inverse_vol(universe, rets),
        strategies.mean_variance(universe, excess),
    ]
    for f in frames:
        assert list(f.index) == list(universe)
        assert np.allclose(f.sum(axis=1), 1.0)
        outside = f.columns.difference(["SPY", "IEF", "TLT", "GLD", "CASH"])
        assert (f[outside] == 0).all().all()
    assert np.allclose(frames[2][["SPY", "IEF"]].iloc[0], [0.6, 0.4])

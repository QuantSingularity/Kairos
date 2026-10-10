import json

import numpy as np
import pandas as pd
import pytest
from kairos import config
from kairos.backtest import engine, holdout
from kairos.backtest import strategies as st
from kairos.data import loader
from kairos.models import black_litterman as bl
from kairos.models import ensembles, pooled, skill, walk_forward
from tests.fixtures import make_forecasts, make_panel, write_raw_cache


def test_calibration_is_bounded_between_base_and_prediction():
    rng = np.random.default_rng(0)
    n = 90
    y = pd.Series(rng.normal(0.01, 0.04, n))
    pred = pd.Series(y.values + rng.normal(0, 0.03, n))
    base = pd.Series(0.01, index=y.index)
    out, slope = ensembles.calibrate_to_mean(pred, y, base)
    ok = slope.dropna()
    assert ((ok >= 0) & (ok <= 1)).all()
    lo, hi = np.minimum(pred, base), np.maximum(pred, base)
    assert ((out >= lo - 1e-12) & (out <= hi + 1e-12)).all()
    assert (slope.iloc[: config.CALIBRATION_MIN] == 0).all()


def test_calibration_shrinks_noise_but_keeps_signal():
    rng = np.random.default_rng(1)
    n = 160
    y = pd.Series(rng.normal(0.0, 0.04, n))
    base = pd.Series(0.0, index=y.index)
    noise, signal = pd.Series(rng.normal(0, 0.05, n)), y + rng.normal(0, 0.01, n)
    _, s_noise = ensembles.calibrate_to_mean(noise, y, base)
    _, s_signal = ensembles.calibrate_to_mean(signal, y, base)
    assert s_noise.iloc[60:].mean() < 0.3 < 0.7 < s_signal.iloc[60:].mean()


def test_calibration_never_uses_future_values():
    rng = np.random.default_rng(2)
    n = 80
    y = pd.Series(rng.normal(0, 0.04, n))
    pred = pd.Series(rng.normal(0, 0.04, n))
    base = pd.Series(0.0, index=y.index)
    a, _ = ensembles.calibrate_to_mean(pred, y, base)
    y2, p2 = y.copy(), pred.copy()
    y2.iloc[50:], p2.iloc[51:] = 9.0, 9.0
    b, _ = ensembles.calibrate_to_mean(p2, y2, base)
    assert np.allclose(a.iloc[:51], b.iloc[:51])


def test_stack_with_fallback_and_weights():
    rng = np.random.default_rng(3)
    n = 60
    y = pd.Series(rng.normal(0, 0.03, n))
    P = pd.DataFrame({"a": y + rng.normal(0, 0.01, n), "b": rng.normal(0, 0.03, n)})
    fallback = pd.Series(0.123, index=y.index)
    _, stacked, w = ensembles.stack_forecasts(
        P, y, window=24, fallback=fallback, return_weights=True
    )
    assert np.allclose(stacked.iloc[:24], 0.123)
    assert (w.dropna() >= 0).all().all()
    assert w.dropna()["a"].mean() > w.dropna()["b"].mean()


def test_lstm_forecasts_are_not_over_dispersed(monkeypatch):
    monkeypatch.setattr(config, "LSTM_EPOCHS", 6)
    rng = np.random.default_rng(4)
    n = 110
    X = rng.normal(size=(n, 4))
    y = rng.normal(0.01, 0.03, n)
    pred = walk_forward.walk_forward_lstm(X, y, start=50)
    ok = ~np.isnan(pred)
    assert 0 < np.std(pred[ok]) / np.std(y[ok]) < 1.5


def synthetic_panel(n=120, seed=5):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2005-01-31", periods=n, freq="ME")
    F = pd.DataFrame(
        {
            "A_mom": rng.normal(size=n),
            "A_mom_12_1": rng.normal(size=n),
            "B_mom": rng.normal(size=n),
            "B_mom_12_1": rng.normal(size=n),
            "VIX_level": rng.normal(size=n),
            "regime": rng.integers(0, 2, n).astype(float),
            "p_stressed": rng.random(n),
        },
        index=idx,
    )
    lp = pd.DataFrame(
        {a: np.cumsum(rng.normal(0.005, 0.04, n)) for a in ("A", "B")}, index=idx
    )
    return F, lp


def test_pooled_forecasts_are_causal_and_start_on_time():
    F, lp = synthetic_panel()
    start = 40
    out = pooled.pooled_forecasts(F, lp, ["A", "B"], start=start)
    assert out["A"]["pooled_ridge"].first_valid_index() == F.index[start]
    t0 = 80
    lp2 = lp.copy()
    lp2.iloc[t0 + 1 :] += 5.0
    F2 = F.copy()
    F2.iloc[t0 + 1 :, :] = 7.0
    alt = pooled.pooled_forecasts(F2, lp2, ["A", "B"], start=start)
    for a in ("A", "B"):
        assert np.allclose(
            out[a].iloc[: t0 + 1].values, alt[a].iloc[: t0 + 1].values, equal_nan=True
        )


def test_prior_weights_variants():
    hist = pd.DataFrame(
        np.random.default_rng(6).normal(0, [0.01, 0.04], (50, 2)),
        columns=["SPY", "EFA"],
    )
    names = ["SPY", "EFA"]
    d = pd.Timestamp("2020-01-31")
    assert bl.prior_weights("equal", names, hist, d) == {"SPY": 1.0, "EFA": 1.0}
    ref = bl.prior_weights("reference", names, hist, d)
    assert ref["SPY"] > ref["EFA"]
    iv = bl.prior_weights("inverse_vol", names, hist, d)
    assert iv["SPY"] > iv["EFA"]
    table = pd.DataFrame(
        {"SPY": [0.3, 0.7], "EFA": [0.7, 0.3]},
        index=pd.to_datetime(["2019-12-31", "2020-01-15"]),
    )
    obs = bl.prior_weights("observed", names, hist, d, table)
    assert obs == {"SPY": 0.7, "EFA": 0.3}
    with pytest.raises(ValueError):
        bl.prior_weights("observed", names, hist, d)
    with pytest.raises(ValueError):
        bl.prior_weights("nonsense", names, hist, d)


def test_prior_portfolio_rows_sum_to_one_and_cover_universe_only():
    rng = np.random.default_rng(7)
    idx = pd.date_range("2015-01-31", periods=60, freq="ME")
    rets = pd.DataFrame(
        rng.normal(0, 0.03, (60, 3)), index=idx, columns=["SPY", "IEF", "GLD"]
    )
    universe = {d: ["SPY", "IEF", "GLD"] for d in idx[40:]}
    for kind in ("reference", "equal", "inverse_vol"):
        w = st.prior_portfolio(universe, kind, rets)
        assert np.allclose(w.sum(axis=1), 1.0)
        assert (w.drop(columns=["SPY", "IEF", "GLD"]) == 0).all().all()


def test_bl_prior_changes_portfolio():
    from dataclasses import replace

    forecasts, rets, rf = make_forecasts()
    u = bl.build_universe(forecasts, rets)
    regime = pd.Series(0.0, index=rets.index)
    base = bl.black_litterman_weights(forecasts, rets, rf, regime, u)
    inv = bl.black_litterman_weights(
        forecasts,
        rets,
        rf,
        regime,
        u,
        params=replace(bl.default_params(), prior="inverse_vol"),
    )
    assert not np.allclose(base.values, inv.values)


def test_spec_hash_and_freeze(tmp_path, monkeypatch):
    paths = config.Paths(tmp_path)
    paths.ensure()
    idx = pd.date_range("2020-01-31", periods=30, freq="ME")
    rng = np.random.default_rng(8)
    ret = pd.DataFrame(
        {"A": rng.normal(0.005, 0.03, 30), "B": rng.normal(0.004, 0.02, 30)}, index=idx
    )
    ret.iloc[:24].to_csv(paths.processed / "backtest_returns.csv")
    pd.DataFrame({"rf": 0.001}, index=idx[:24]).to_csv(
        paths.processed / "backtest_rf.csv"
    )
    first = holdout.freeze(paths)
    assert first["freeze_date"] == str(idx[23].date())
    assert holdout.freeze(paths) == first
    status = holdout.evaluate(paths).iloc[0]
    assert status["months_since_freeze"] == 0 and status["spec_unchanged"]
    assert not (paths.tables / "holdout_metrics.csv").exists()
    ret.to_csv(paths.processed / "backtest_returns.csv")
    pd.DataFrame({"rf": 0.001}, index=idx).to_csv(paths.processed / "backtest_rf.csv")
    status = holdout.evaluate(paths).iloc[0]
    assert status["months_since_freeze"] == 6 and not status["inference_possible"]
    assert len(pd.read_csv(paths.tables / "holdout_metrics.csv")) == 2
    h = holdout.spec_hash(paths)
    monkeypatch.setattr(config, "TAU", 0.123)
    assert holdout.spec_hash(paths) != h
    assert not holdout.evaluate(paths).iloc[0]["spec_unchanged"]
    assert (
        json.loads((tmp_path / "results" / "freeze.json").read_text())["spec_hash"]
        == first["spec_hash"]
    )


def test_period_tables_split_at_cut(monkeypatch):
    monkeypatch.setattr(config, "HOLDOUT_START", "2016-01-01")
    idx = pd.date_range("2013-01-31", periods=60, freq="ME")
    rng = np.random.default_rng(9)

    def bt(mu):
        r = rng.normal(mu, 0.03, 60)
        return pd.DataFrame({"net": r, "rf": 0.001, "turnover": 0.1}, index=idx)

    main = {"BL_ML_Regime": bt(0.005), "Static_60_40": bt(0.004)}
    excess = {n: b["net"] - b["rf"] for n, b in main.items()}
    table, tests = engine.period_tables(main, excess)
    assert table.loc[("BL_ML_Regime", "development"), "Months"] == 36
    assert table.loc[("BL_ML_Regime", "holdout"), "Months"] == 24
    assert set(tests["period"]) == {"development", "holdout"}


def test_skill_by_period_counts():
    rng = np.random.default_rng(10)
    n = 80
    idx = pd.date_range("2013-01-31", periods=n, freq="ME")
    target = rng.normal(0.01, 0.04, n)
    df = pd.DataFrame(
        {c: target + rng.normal(0, 0.05, n) for c in skill.SKILL_COLUMNS}, index=idx
    )
    df["target"] = target
    table = skill.skill_by_period({"X": df}, cut="2016-01-01")
    assert set(table["Period"]) == {"full", "development", "holdout"}
    assert (table["Assets"] == 1).all()


def test_data_manifest_lists_cached_series(tmp_path):
    paths = config.Paths(tmp_path)
    paths.ensure()
    write_raw_cache(paths, make_panel(start="2004-01-01", end="2010-12-31"))
    m = loader.manifest(paths)
    assert {"SPY", "^VIX"} <= set(m["series"])
    assert (m["rows"] > 100).all() and m["sha256"].str.len().eq(16).all()

import numpy as np
import pandas as pd
from kairos import config
from kairos.backtest import stats, strategies
from kairos.models import black_litterman as bl
from kairos.models import dataset
from kairos.models import diagnostics as dg
from kairos.models import skill
from tests.fixtures import make_forecasts


def test_deflated_sharpe_single_trial_has_zero_hurdle():
    x = np.random.default_rng(0).normal(0.01, 0.03, 200)
    r = stats.deflated_sharpe(x, 1, 0.0)
    assert r["BenchmarkSharpe"] == 0.0 and 0.5 < r["DSR"] <= 1.0


def test_deflated_sharpe_falls_with_more_trials():
    x = np.random.default_rng(1).normal(0.006, 0.03, 200)
    few = stats.deflated_sharpe(x, 3, 0.002)
    many = stats.deflated_sharpe(x, 50, 0.002)
    assert many["BenchmarkSharpe"] > few["BenchmarkSharpe"]
    assert many["DSR"] < few["DSR"]


def test_deflated_sharpe_separates_signal_from_noise():
    rng = np.random.default_rng(2)
    good = stats.deflated_sharpe(rng.normal(0.02, 0.02, 200), 3, 0.001)
    bad = stats.deflated_sharpe(rng.normal(0.0, 0.03, 200), 3, 0.001)
    assert good["DSR"] > 0.99 and bad["DSR"] < 0.9


def test_regime_overlay_cuts_risk_only_when_stressed():
    idx = pd.date_range("2015-01-31", periods=6, freq="ME")
    w = pd.DataFrame(0.0, index=idx, columns=config.ASSETS + ["CASH"])
    w["SPY"], w["IEF"] = 0.6, 0.4
    regime = pd.Series([0.0, 1.0, 1.0, np.nan, 0.0, 1.0], index=idx)
    out = strategies.regime_overlay(w, regime, scale=0.5)
    assert np.allclose(out.sum(axis=1), 1.0)
    stressed = (regime == 1.0).values
    assert np.allclose(out.loc[stressed, "SPY"], 0.3)
    assert np.allclose(out.loc[stressed, "CASH"], 0.5)
    assert np.allclose(out.loc[~stressed, "SPY"], 0.6)
    assert np.allclose(out.loc[~stressed, "CASH"], 0.0)


def test_default_overlay_scale_matches_risk_aversion_multiplier():
    idx = pd.date_range("2015-01-31", periods=2, freq="ME")
    w = pd.DataFrame(0.0, index=idx, columns=config.ASSETS + ["CASH"])
    w["SPY"] = 1.0
    out = strategies.regime_overlay(w, pd.Series(1.0, index=idx))
    assert np.allclose(out["SPY"], 1.0 / config.STRESSED_DELTA_MULT)


def test_bl_params_default_to_config_and_change_weights():
    assert bl.default_params().tau == config.TAU
    forecasts, rets, rf = make_forecasts()
    u = bl.build_universe(forecasts, rets)
    regime = pd.Series(0.0, index=rets.index)
    base = bl.black_litterman_weights(forecasts, rets, rf, regime, u)
    same = bl.black_litterman_weights(
        forecasts, rets, rf, regime, u, params=bl.default_params()
    )
    assert np.allclose(base.values, same.values)
    from dataclasses import replace

    for change in ({"tau": 0.5}, {"prior": "equal"}, {"max_weight": 0.2}):
        alt = bl.black_litterman_weights(
            forecasts,
            rets,
            rf,
            regime,
            u,
            params=replace(bl.default_params(), **change),
        )
        assert not np.allclose(base.values, alt.values)
        assert np.allclose(alt.sum(axis=1), 1.0)
    capped = bl.black_litterman_weights(
        forecasts,
        rets,
        rf,
        regime,
        u,
        params=replace(bl.default_params(), max_weight=0.2),
    )
    assert (capped.drop(columns="CASH") <= 0.2 + 1e-9).all().all()


def test_historical_mean_is_causal():
    y = pd.Series([1.0, 3.0, 5.0, 7.0])
    hm = dataset.historical_mean(y)
    assert np.isnan(hm.iloc[0]) and hm.iloc[1] == 1.0 and hm.iloc[3] == 3.0
    y2 = y.copy()
    y2.iloc[3] = 100.0
    assert np.allclose(
        dataset.historical_mean(y2).iloc[:3].dropna(), hm.iloc[:3].dropna()
    )


def test_r2_oos_reference_values():
    target = np.array([1.0, -1.0, 2.0, 0.5])
    assert np.isclose(skill.r2_oos(target, target, np.zeros(4)), 1.0)
    assert np.isclose(skill.r2_oos(np.zeros(4), target, np.zeros(4)), 0.0)
    assert skill.r2_oos(target * 3, target, np.zeros(4)) < 0


def make_forecast_frame(n=100, seed=0, always_up=False):
    rng = np.random.default_rng(seed)
    target = rng.normal(0.01, 0.04, n)
    cols = config.MODELS + ["ensemble_equal", "ensemble_stacked"]
    pred = np.full(n, 0.02) if always_up else target + rng.normal(0, 0.04, n)
    df = pd.DataFrame({c: pred for c in cols})
    df["target"] = target
    df.index = pd.date_range("2012-01-31", periods=n, freq="ME")
    return df


def test_evaluate_reports_base_rate_and_hit_edge():
    df = make_forecast_frame(always_up=True)
    row = skill.evaluate({"X": df}).iloc[0]
    assert np.isclose(row["Hit_ensemble_stacked"], row["BaseRate"])
    assert np.isclose(row["HitEdge_ensemble_stacked"], 0.0)
    assert row["UpShare_stacked"] == 1.0


def test_evaluate_r2_and_dispersion_flag_overscaled_forecasts():
    df = make_forecast_frame()
    good = skill.evaluate({"X": df}).iloc[0]
    wild = df.copy()
    for c in config.MODELS:
        wild[c] = wild[c] * 10
    bad = skill.evaluate({"X": wild}).iloc[0]
    assert bad["R2OOS_xgb"] < good["R2OOS_xgb"]
    assert bad["Disp_xgb"] > 5 * good["Disp_xgb"] * 0.9


def synthetic_features(n=130, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2005-01-31", periods=n, freq="ME")
    ret = rng.normal(0.005, 0.04, n)
    log_price = pd.Series(np.cumsum(ret), index=idx)
    nxt = log_price.shift(-1) - log_price
    p = np.where(np.arange(n) < 48, -nxt.fillna(0).values, nxt.fillna(0).values)
    p = p + rng.normal(0, 0.01, n)
    F = pd.DataFrame(
        {
            "SPY_mom_12_1": rng.normal(size=n),
            "SPY_vol_21d": rng.normal(size=n),
            "VIX_level": rng.normal(size=n),
            "regime": (p > np.median(p)).astype(float),
            "p_stressed": p,
        },
        index=idx,
    )
    return F, log_price


def test_regime_return_link_detects_sign_flip(monkeypatch):
    monkeypatch.setattr(config, "INITIAL_TRAIN_MONTHS", 48)
    F, lp = synthetic_features()
    table = dg.regime_return_link(F, pd.DataFrame({"SPY": lp}), ["SPY"]).set_index(
        "Window"
    )
    assert table.at["initial_window", "CorrStressNextRet"] < -0.5
    assert table.at["after", "CorrStressNextRet"] > 0.5


def test_variant_forecast_ignores_regime_columns_when_dropped(monkeypatch):
    monkeypatch.setattr(config, "DIAG_MODELS", ["ridge"])
    F, lp = synthetic_features()
    kept = dg.variant_forecast("SPY", F, lp, (), 70)
    dropped = dg.variant_forecast("SPY", F, lp, dg.REGIME_FEATURES, 70)
    F2 = F.copy()
    F2["regime"], F2["p_stressed"] = (
        F["regime"].values[::-1],
        F["p_stressed"].values[::-1],
    )
    dropped2 = dg.variant_forecast("SPY", F2, lp, dg.REGIME_FEATURES, 70)
    kept2 = dg.variant_forecast("SPY", F2, lp, (), 70)
    assert np.allclose(dropped["ridge"], dropped2["ridge"], equal_nan=True)
    assert not np.allclose(kept["ridge"], kept2["ridge"], equal_nan=True)
    assert dropped["ridge"].first_valid_index() == dropped.index[70]


def test_evaluate_variants_compares_on_common_dates(monkeypatch):
    monkeypatch.setattr(config, "DIAG_MODELS", ["ridge"])
    rng = np.random.default_rng(4)
    idx = pd.date_range("2010-01-31", periods=90, freq="ME")
    target = rng.normal(0.01, 0.04, 90)
    a = pd.DataFrame(
        {"target": target, "ridge": target + rng.normal(0, 0.04, 90)}, index=idx
    )
    b = a.copy()
    a.iloc[:20, 1] = np.nan
    b.iloc[:40, 1] = np.nan
    frames = pd.concat(
        [
            a.assign(variant="as_specified", asset="X"),
            b.assign(variant="longer_initial_window", asset="X"),
        ]
    )
    table = dg.evaluate_variants(frames)
    assert set(table["Variant"]) == {"as_specified", "longer_initial_window"}
    assert (table["n_oos"] == 50).all()
    summary = dg.summarise_variants(table)
    assert list(summary.index) == ["as_specified", "longer_initial_window"]

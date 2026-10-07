import numpy as np
import pandas as pd
from fixtures import make_panel

from models import regime_detection as rd


def synthetic_inputs(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    state = np.zeros(n, dtype=int)
    for t in range(1, n):
        state[t] = state[t - 1] if rng.random() > 0.03 else 1 - state[t - 1]
    vol = np.where(state == 1, 0.35, 0.10)
    ret = rng.normal(0, vol / np.sqrt(252))
    idx = pd.bdate_range("2010-01-01", periods=n)
    realised = pd.Series(ret, index=idx).rolling(
        21, min_periods=5
    ).std().bfill() * np.sqrt(252)
    df = pd.DataFrame(
        {
            "ret": ret,
            "vol21": realised.values,
            "vix": 12 + 60 * vol + rng.normal(0, 3, n),
        },
        index=idx,
    )
    return df, state


def test_forward_filter_matches_smoothed_at_last_point():
    df, _ = synthetic_inputs(600)
    X = (df.values - df.values.mean(0)) / df.values.std(0)
    model = rd.fit_hmm(X, 2)
    filt = rd.forward_filter(model, X)
    for t in (200, 399, 599):
        assert np.allclose(filt[t], model.predict_proba(X[: t + 1])[-1], atol=1e-6)
    assert np.allclose(filt.sum(axis=1), 1.0)


def test_walk_forward_is_causal():
    df, _ = synthetic_inputs(1200)
    full = rd.walk_forward_hmm(df, min_train=400, refit_every=200)
    cut = 800
    part = rd.walk_forward_hmm(df.iloc[:cut], min_train=400, refit_every=200)
    both = full.iloc[:cut].dropna()
    assert np.allclose(both.values, part.loc[both.index].values, atol=1e-6)
    assert full.iloc[:400].isna().all()


def test_walk_forward_recovers_states():
    df, state = synthetic_inputs(1500)
    p = rd.walk_forward_hmm(df, min_train=500, refit_every=250).dropna()
    truth = state[-len(p) :]
    accuracy = ((p.values > 0.5).astype(int) == truth).mean()
    assert accuracy > 0.85


def test_volatility_regime_and_validation():
    panel = make_panel(start="2007-01-01", end="2013-12-31").dropna(
        subset=["SPY", "VIX"]
    )
    out = rd.detect(panel, min_train=504, refit_every=252)
    assert set(out["regime"].dropna().unique()) <= {0.0, 1.0}
    table = rd.validate(out["regime"])
    stress = table[table["Kind"] == "stress"].dropna(subset=["Rate"])
    assert not stress.empty and stress["Rate"].mean() > 0.5
    assert set(table["Kind"]) == {"stress", "calm"}

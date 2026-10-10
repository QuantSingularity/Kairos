import numpy as np
import pandas as pd
from kairos import config
from kairos.models import dataset, ensembles, skill, walk_forward


def test_targets_are_next_month_return():
    lp = pd.Series(
        np.log([100.0, 110.0, 121.0]),
        index=pd.date_range("2020-01-31", periods=3, freq="ME"),
    )
    t = dataset.forward_targets(lp)
    assert np.isclose(t.iloc[0], np.log(1.1)) and np.isnan(t.iloc[-1])


def make_xy(n=110, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    y = 0.02 * X[:, 0] + rng.normal(0, 0.01, n)
    return X, y


def test_walk_forward_sklearn_never_uses_future():
    X, y = make_xy()
    base = walk_forward.walk_forward_sklearn(X, y, ["ridge"], start=60)["ridge"]
    t = 90
    X2, y2 = X.copy(), y.copy()
    X2[t + 1 :] = 99.0
    y2[t:] = 99.0
    alt = walk_forward.walk_forward_sklearn(X2, y2, ["ridge"], start=60)["ridge"]
    assert np.allclose(base[: t + 1], alt[: t + 1], equal_nan=True)
    assert np.isnan(base[:60]).all()


def test_walk_forward_lstm_never_uses_future():
    X, y = make_xy(n=100)
    old = (config.INITIAL_TRAIN_MONTHS, config.LSTM_REFIT_EVERY)
    base = walk_forward.walk_forward_lstm(X, y, start=55)
    t = 70
    X2, y2 = X.copy(), y.copy()
    X2[t + 1 :] = 50.0
    y2[t:] = 50.0
    alt = walk_forward.walk_forward_lstm(X2, y2, start=55)
    assert np.allclose(base[: t + 1], alt[: t + 1], equal_nan=True)
    assert (config.INITIAL_TRAIN_MONTHS, config.LSTM_REFIT_EVERY) == old


def test_stacking_uses_only_past_and_is_nonnegative():
    rng = np.random.default_rng(1)
    n = 60
    y = pd.Series(rng.normal(0, 0.03, n))
    P = pd.DataFrame({"a": y + rng.normal(0, 0.01, n), "b": rng.normal(0, 0.03, n)})
    equal, stacked = ensembles.stack_forecasts(P, y, window=24)
    P2, y2 = P.copy(), y.copy()
    P2.iloc[41:] = 7.0
    y2.iloc[40:] = 7.0
    _, stacked2 = ensembles.stack_forecasts(P2, y2, window=24)
    assert np.allclose(stacked.iloc[:41], stacked2.iloc[:41])
    assert np.allclose(equal, P.mean(axis=1))
    assert np.corrcoef(stacked.iloc[30:], y.iloc[30:])[0, 1] > 0.5


def test_evaluate_reports_ic():
    rng = np.random.default_rng(2)
    n = 80
    target = rng.normal(size=n)
    cols = config.MODELS + ["ensemble_equal", "ensemble_stacked"]
    df = pd.DataFrame({c: target + rng.normal(0, 1, n) for c in cols})
    df["target"] = target
    df.index = pd.date_range("2015-01-31", periods=n, freq="ME")
    table = skill.evaluate({"X": df})
    assert table.loc[0, "n_oos"] == n
    assert table.loc[0, "IC_ensemble_stacked"] > 0.3
    assert table.loc[0, "BestSingle"] in config.MODELS

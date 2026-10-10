import numpy as np
import pandas as pd

from kairos import config
from kairos.models import black_litterman as bl
from tests.fixtures import make_forecasts


def toy(n=4, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    cov = pd.DataFrame(
        A @ A.T / 400 + np.eye(n) * 0.0004,
        columns=list("abcd")[:n],
        index=list("abcd")[:n],
    )
    return cov


def test_posterior_limits():
    cov = toy()
    pi = np.array([0.004, 0.003, 0.002, 0.001])
    Q = np.array([0.02, -0.01, 0.03, 0.0])
    mu_far, _ = bl.posterior(cov, pi, Q, np.full(4, 1e9))
    mu_tight, _ = bl.posterior(cov, pi, Q, np.full(4, 1e-12))
    assert np.allclose(mu_far, pi, atol=1e-6)
    assert np.allclose(mu_tight, Q, atol=1e-5)


def test_posterior_is_between_prior_and_view():
    cov = toy()
    pi, Q = np.full(4, 0.002), np.full(4, 0.010)
    mu, cov_bl = bl.posterior(cov, pi, Q, np.full(4, cov.values.diagonal().mean()))
    assert (mu > 0.002).all() and (mu < 0.010).all()
    assert (np.linalg.eigvalsh(cov_bl) > 0).all()


def test_equilibrium_reverse_optimisation_roundtrip():
    cov = toy()
    w = {"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1}
    pi = bl.equilibrium_returns(cov, w, 2.5)
    recovered = np.linalg.solve(2.5 * cov.values, pi)
    assert np.allclose(recovered, [0.4, 0.3, 0.2, 0.1])


def test_optimizer_constraints():
    cov = toy()
    mu = np.array([0.01, 0.008, 0.006, -0.01])
    w = bl.optimize_weights(mu, cov.values, 2.5)
    assert (
        (w >= -1e-9).all()
        and (w <= config.MAX_WEIGHT + 1e-9).all()
        and w.sum() <= 1 + 1e-9
    )
    assert w[3] < 1e-6


def test_higher_risk_aversion_raises_cash():
    cov = toy()
    mu = np.array([0.006, 0.005, 0.004, 0.003])
    low = bl.optimize_weights(mu, cov.values, 2.5)
    high = bl.optimize_weights(mu, cov.values, 2.5 * config.STRESSED_DELTA_MULT)
    assert high.sum() < low.sum() - 1e-4


def test_optimizer_matches_closed_form_when_unconstrained():
    cov = pd.DataFrame(np.diag([0.01, 0.02, 0.03, 0.04]))
    mu = np.array([0.004, 0.003, 0.0035, 0.0032])
    delta = 40.0
    w = bl.optimize_weights(mu, cov.values, delta, max_weight=1.0)
    assert np.allclose(w, np.linalg.solve(delta * cov.values, mu), atol=1e-5)


def test_universe_requires_history_and_forecasts():
    forecasts, rets, rf = make_forecasts()
    u = bl.build_universe(forecasts, rets)
    assert min(u) >= forecasts["SPY"]["ensemble_final"].dropna().index[0]
    assert all(len(v) >= config.MIN_UNIVERSE for v in u.values())


def test_weights_valid_and_regime_effect():
    forecasts, rets, rf = make_forecasts()
    u = bl.build_universe(forecasts, rets)
    stressed = pd.Series(1.0, index=rets.index)
    calm = pd.Series(0.0, index=rets.index)
    ws = bl.black_litterman_weights(forecasts, rets, rf, stressed, u, use_regime=True)
    wc = bl.black_litterman_weights(forecasts, rets, rf, calm, u, use_regime=True)
    wn = bl.black_litterman_weights(forecasts, rets, rf, stressed, u, use_regime=False)
    for w in (ws, wc, wn):
        assert np.allclose(w.sum(axis=1), 1.0)
        assert (w >= -1e-9).all().all()
    assert ws["CASH"].mean() > wc["CASH"].mean()
    assert np.allclose(wn.values, wc.values)

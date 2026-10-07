import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

import config

COLUMNS = config.ASSETS + ["CASH"]


def shrunk_cov(returns):
    cov = LedoitWolf().fit(returns.values).covariance_
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


def equilibrium_returns(cov, weights, delta):
    w = np.array([weights[a] for a in cov.columns], dtype=float)
    return delta * cov.values @ (w / w.sum())


def posterior(cov, pi, Q, omega_diag, tau=config.TAU):
    sigma_inv = np.linalg.inv(tau * cov.values)
    omega_inv = np.diag(1.0 / np.asarray(omega_diag))
    M = np.linalg.inv(sigma_inv + omega_inv)
    return M @ (sigma_inv @ pi + omega_inv @ Q), cov.values + M


def optimize_weights(mu, cov, delta, max_weight=config.MAX_WEIGHT):
    mu, cov, n = np.asarray(mu, dtype=float), np.asarray(cov, dtype=float), len(mu)
    res = minimize(
        lambda w: -(mu @ w - 0.5 * delta * w @ cov @ w),
        np.full(n, min(max_weight, 1.0 / n)),
        jac=lambda w: -(mu - delta * cov @ w),
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n,
        constraints=[
            {
                "type": "ineq",
                "fun": lambda w: 1.0 - w.sum(),
                "jac": lambda w: -np.ones(n),
            }
        ],
        options={"maxiter": 200, "ftol": 1e-12},
    )
    w = np.clip(res.x, 0.0, max_weight)
    return w / w.sum() if w.sum() > 1.0 else w


def build_universe(
    forecasts,
    log_rets,
    min_history=config.MIN_HISTORY,
    min_universe=config.MIN_UNIVERSE,
):
    dates = sorted(
        set().union(
            *[set(df["ensemble_stacked"].dropna().index) for df in forecasts.values()]
        )
    )
    history = log_rets.notna().cumsum()
    universe = {}
    for d in dates:
        names = [
            a
            for a, df in forecasts.items()
            if d in df.index
            and not np.isnan(df.at[d, "ensemble_stacked"])
            and history.at[d, a] >= min_history
        ]
        if len(names) >= min_universe:
            universe[d] = names
    return universe


def trailing_stats(df, date, window=config.STACK_WINDOW):
    hist = df.loc[df.index < date, ["target", "ensemble_stacked"]].dropna().tail(window)
    err_var = (
        float(max((hist["ensemble_stacked"] - hist["target"]).var(), 1e-6))
        if len(hist) >= 6
        else None
    )
    ic = hist["ensemble_stacked"].corr(hist["target"]) if len(hist) >= 8 else 0.0
    return err_var, 0.0 if np.isnan(ic) else float(ic)


def black_litterman_weights(
    forecasts, log_rets, rf, regime, universe, use_regime=True, ic_shrink=False
):
    excess = log_rets.sub(rf, axis=0)
    records = {}
    for d, names in universe.items():
        hist = excess.loc[:d, names].dropna().tail(config.COV_WINDOW)
        cov = shrunk_cov(hist)
        stressed = use_regime and regime.get(d, 0.0) == 1.0
        delta = config.BASE_DELTA * (config.STRESSED_DELTA_MULT if stressed else 1.0)
        pi = equilibrium_returns(cov, config.MARKET_WEIGHTS, config.BASE_DELTA)
        Q = np.array([forecasts[a].at[d, "ensemble_stacked"] - rf.at[d] for a in names])
        omega = []
        for i, a in enumerate(names):
            err_var, ic = trailing_stats(forecasts[a], d)
            base = err_var if err_var is not None else float(np.var(hist[a]))
            mult = config.STRESSED_OMEGA_MULT if stressed else 1.0
            if ic_shrink:
                mult /= max(ic, config.IC_FLOOR)
            omega.append(base * mult)
        mu, cov_bl = posterior(cov, pi, Q, omega)
        w = optimize_weights(mu, cov_bl, delta)
        row = pd.Series(w, index=names)
        row["CASH"] = 1.0 - w.sum()
        records[d] = row
    return pd.DataFrame(records).T.reindex(columns=COLUMNS).fillna(0.0)

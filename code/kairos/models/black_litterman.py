from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from kairos import config

COLUMNS = config.ASSETS + ["CASH"]


@dataclass(frozen=True)
class BLParams:
    tau: float
    base_delta: float
    stressed_delta_mult: float
    stressed_omega_mult: float
    max_weight: float
    prior: str = "reference"


def default_params():
    return BLParams(
        config.TAU,
        config.BASE_DELTA,
        config.STRESSED_DELTA_MULT,
        config.STRESSED_OMEGA_MULT,
        config.MAX_WEIGHT,
    )


def load_prior_table(paths=config.PATHS):
    path = paths.raw / config.OBSERVED_PRIOR_FILE
    if not path.exists():
        return None
    table = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return table[[c for c in table.columns if c in config.ASSETS]]


def prior_weights(kind, names, hist, date, table=None):
    if kind == "reference":
        return {a: config.MARKET_WEIGHTS[a] for a in names}
    if kind == "equal":
        return dict.fromkeys(names, 1.0)
    if kind == "inverse_vol":
        return (1.0 / hist[names].std().replace(0, np.nan)).fillna(1.0).to_dict()
    if kind == "observed":
        if table is None or table.loc[:date].empty:
            raise ValueError(
                "the observed prior needs a weight table covering the date"
            )
        row = table.loc[:date].iloc[-1]
        w = {a: float(np.nan_to_num(row.get(a, 0.0))) for a in names}
        if sum(w.values()) <= 0:
            raise ValueError("observed prior weights are empty for the eligible assets")
        return w
    raise ValueError(f"unknown prior {kind}")


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
    view=None,
):
    view = view or config.VIEW_COLUMN
    dates = sorted(
        set().union(*[set(df[view].dropna().index) for df in forecasts.values()])
    )
    history = log_rets.notna().cumsum()
    universe = {}
    for d in dates:
        names = [
            a
            for a, df in forecasts.items()
            if d in df.index
            and not np.isnan(df.at[d, view])
            and history.at[d, a] >= min_history
        ]
        if len(names) >= min_universe:
            universe[d] = names
    return universe


def trailing_stats(df, date, window=config.STACK_WINDOW, view=None):
    view = view or config.VIEW_COLUMN
    hist = df.loc[df.index < date, ["target", view]].dropna().tail(window)
    err_var = (
        float(max((hist[view] - hist["target"]).var(), 1e-6))
        if len(hist) >= 6
        else None
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        ic = hist[view].corr(hist["target"]) if len(hist) >= 8 else 0.0
    return err_var, 0.0 if np.isnan(ic) else float(ic)


def black_litterman_weights(
    forecasts,
    log_rets,
    rf,
    regime,
    universe,
    use_regime=True,
    ic_shrink=False,
    params=None,
    view=None,
    prior_table=None,
):
    p = params or default_params()
    view = view or config.VIEW_COLUMN
    excess = log_rets.sub(rf, axis=0)
    records = {}
    for d, names in universe.items():
        hist = excess.loc[:d, names].dropna().tail(config.COV_WINDOW)
        cov = shrunk_cov(hist)
        stressed = use_regime and regime.get(d, 0.0) == 1.0
        delta = p.base_delta * (p.stressed_delta_mult if stressed else 1.0)
        prior = prior_weights(p.prior, names, hist, d, prior_table)
        pi = equilibrium_returns(cov, prior, p.base_delta)
        Q = np.array([forecasts[a].at[d, view] - rf.at[d] for a in names])
        omega = []
        for i, a in enumerate(names):
            err_var, ic = trailing_stats(forecasts[a], d, view=view)
            base = err_var if err_var is not None else float(np.var(hist[a]))
            mult = p.stressed_omega_mult if stressed else 1.0
            if ic_shrink:
                mult /= max(ic, config.IC_FLOOR)
            omega.append(base * mult)
        mu, cov_bl = posterior(cov, pi, Q, omega, p.tau)
        w = optimize_weights(mu, cov_bl, delta, p.max_weight)
        row = pd.Series(w, index=names)
        row["CASH"] = 1.0 - w.sum()
        records[d] = row
    return pd.DataFrame(records).T.reindex(columns=COLUMNS).fillna(0.0)

import numpy as np
import pandas as pd

import config
from models.black_litterman import COLUMNS, optimize_weights, shrunk_cov


def to_frame(records):
    return pd.DataFrame(records).T.reindex(columns=COLUMNS).fillna(0.0)


def equal_weight(universe):
    return to_frame({d: pd.Series(1.0 / len(a), index=a) for d, a in universe.items()})


def market_reference(universe):
    out = {}
    for d, a in universe.items():
        w = pd.Series({x: config.MARKET_WEIGHTS[x] for x in a})
        out[d] = w / w.sum()
    return to_frame(out)


def static_60_40(universe):
    return to_frame({d: pd.Series({"SPY": 0.6, "IEF": 0.4}) for d in universe})


def inverse_vol(universe, log_rets, window=36):
    out = {}
    for d, a in universe.items():
        vol = log_rets.loc[:d, a].dropna().tail(window).std()
        inv = 1.0 / vol.replace(0, np.nan)
        out[d] = (inv / inv.sum()).fillna(1.0 / len(a))
    return to_frame(out)


def mean_variance(universe, excess, window=36):
    out = {}
    for d, a in universe.items():
        hist = excess.loc[:d, a].dropna().tail(window)
        w = optimize_weights(
            hist.mean().values, shrunk_cov(hist).values, config.BASE_DELTA
        )
        row = pd.Series(w, index=a)
        row["CASH"] = 1.0 - w.sum()
        out[d] = row
    return to_frame(out)


def regime_overlay(weights, regime, scale=None):
    scale = 1.0 / config.STRESSED_DELTA_MULT if scale is None else scale
    out = weights.copy()
    risky = [c for c in out.columns if c != "CASH"]
    stressed = (regime.reindex(out.index) == 1.0).values
    out.loc[stressed, risky] = out.loc[stressed, risky] * scale
    out["CASH"] = 1.0 - out[risky].sum(axis=1)
    return out


def prior_portfolio(universe, kind, log_rets, table=None):
    from models.black_litterman import prior_weights

    out = {}
    for d, names in universe.items():
        hist = log_rets.loc[:d, names].dropna().tail(config.COV_WINDOW)
        w = pd.Series(prior_weights(kind, names, hist, d, table), dtype=float)
        out[d] = w / w.sum()
    return to_frame(out)

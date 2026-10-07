import numpy as np
import pandas as pd

import config

LAUNCH = {
    "HYG": "2007-04-11",
    "GLD": "2004-11-18",
    "DBC": "2006-02-06",
    "BTC-USD": "2014-09-17",
    "ETH-USD": "2017-11-09",
}
CRISES = [
    ("2008-09-01", "2009-03-31"),
    ("2011-07-15", "2011-10-15"),
    ("2015-08-15", "2016-02-15"),
    ("2018-10-01", "2018-12-24"),
    ("2020-02-19", "2020-04-07"),
    ("2022-01-03", "2022-10-14"),
]


def make_panel(start="2003-01-01", end="2026-09-30", seed=7):
    """Factor-model panel with regime-switching volatility. Used only to exercise the code in tests."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    n = len(idx)
    stress = np.zeros(n, dtype=bool)
    for a, b in CRISES:
        stress |= (idx >= a) & (idx <= b)
    vol = np.where(stress, 0.030, 0.009)
    mkt = rng.normal(np.where(stress, -0.0015, 0.0006), vol)
    rate = rng.normal(0, 0.003, n) - 0.15 * mkt
    load = {
        "SPY": (1.0, 0.0),
        "EFA": (0.9, 0.0),
        "TLT": (-0.15, 1.0),
        "IEF": (-0.1, 0.45),
        "HYG": (0.45, 0.1),
        "GLD": (0.0, 0.2),
        "DBC": (0.5, -0.1),
        "BTC-USD": (0.9, 0.0),
        "ETH-USD": (1.0, 0.0),
    }
    idio = {
        "SPY": 0.002,
        "EFA": 0.004,
        "TLT": 0.0015,
        "IEF": 0.001,
        "HYG": 0.002,
        "GLD": 0.008,
        "DBC": 0.008,
        "BTC-USD": 0.03,
        "ETH-USD": 0.035,
    }
    cols = {}
    for a, (bm, br) in load.items():
        ret = bm * mkt + br * rate + rng.normal(0, idio[a], n)
        s = pd.Series(100 * np.exp(np.cumsum(ret)), index=idx)
        if a in LAUNCH:
            s[s.index < LAUNCH[a]] = np.nan
        cols[a] = s
    panel = pd.DataFrame(cols)
    panel["VIX"] = pd.Series(
        np.where(stress, 38.0, 15.0) + rng.normal(0, 2, n), index=idx
    ).clip(lower=9)
    panel["RF"] = pd.Series(
        np.clip(0.02 + np.cumsum(rng.normal(0, 0.00005, n)), 0.0, 0.05), index=idx
    )
    return panel[[c for c in config.ASSETS] + ["VIX", "RF"]]


def write_raw_cache(paths, panel):
    import data.loader as loader

    paths.raw.mkdir(parents=True, exist_ok=True)
    mapping = {a: a for a in config.ASSETS}
    mapping.update({"VIX": config.VIX_TICKER, "RF": config.RF_TICKER})
    for col, ticker in mapping.items():
        s = panel[col].dropna() * (100.0 if col == "RF" else 1.0)
        s.name = ticker
        s.to_csv(loader.cache_file(paths, ticker), header=True)

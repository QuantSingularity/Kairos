import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

import config


def stationary_indices(n, draws, mean_block, rng):
    restart = rng.random((draws, n)) < 1.0 / mean_block
    restart[:, 0] = True
    starts = rng.integers(0, n, (draws, n))
    idx = np.empty((draws, n), dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    for t in range(1, n):
        idx[:, t] = np.where(restart[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def sharpe(x):
    sd = x.std(axis=-1, ddof=1)
    return np.where(
        sd > 0, x.mean(axis=-1) / np.where(sd > 0, sd, 1.0) * np.sqrt(12), np.nan
    )


def paired_sharpe_test(
    excess_a,
    excess_b,
    draws=config.BOOTSTRAP_DRAWS,
    block=config.BOOTSTRAP_BLOCK,
    seed=config.SEED,
):
    df = pd.concat([excess_a, excess_b], axis=1, keys=["a", "b"]).dropna()
    n = len(df)
    xa, xb = df["a"].values, df["b"].values
    observed = float(sharpe(xa) - sharpe(xb))
    idx = stationary_indices(n, draws, block, np.random.default_rng(seed))
    diffs = sharpe(xa[idx]) - sharpe(xb[idx])
    diffs = diffs[~np.isnan(diffs)]
    p = (np.sum(np.abs(diffs - observed) >= abs(observed)) + 1) / (len(diffs) + 1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "diff": observed,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(p),
        "n": n,
    }


EULER_GAMMA = 0.5772156649015329


def deflated_sharpe(excess, n_trials, sr_var):
    x = np.asarray(excess, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    sr = x.mean() / x.std(ddof=1)
    if n_trials > 1 and sr_var > 0:
        sr0 = np.sqrt(sr_var) * (
            (1 - EULER_GAMMA) * norm.ppf(1 - 1.0 / n_trials)
            + EULER_GAMMA * norm.ppf(1 - 1.0 / (n_trials * np.e))
        )
    else:
        sr0 = 0.0
    denom = np.sqrt(1 - skew(x) * sr + (kurtosis(x, fisher=False) - 1) / 4 * sr**2)
    prob = float(norm.cdf((sr - sr0) * np.sqrt(n - 1) / denom))
    return {
        "Sharpe": float(sr * np.sqrt(12)),
        "BenchmarkSharpe": float(sr0 * np.sqrt(12)),
        "DSR": prob,
        "Trials": n_trials,
    }

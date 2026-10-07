import numpy as np
import pandas as pd

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

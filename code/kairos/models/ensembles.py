"""Stacking and calibration of forecasts."""

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from kairos import config

CONVEX_PENALTY = 1e3


def stack_forecasts(
    P: pd.DataFrame,
    y: pd.Series,
    window: int | None = None,
    fallback: pd.Series | None = None,
    return_weights: bool = False,
    convex: bool = False,
) -> tuple[Any, ...]:
    """Equal-weight and stacked forecasts, optionally with weights."""
    window = config.STACK_WINDOW if window is None else window
    base = P.mean(axis=1, skipna=False) if fallback is None else fallback
    stacked = base.copy()
    values, target = P.values, y.values
    weights = pd.DataFrame(np.nan, index=P.index, columns=P.columns)
    for i in range(len(P)):
        if np.isnan(values[i]).any():
            continue
        hist = np.arange(i)
        ok = hist[~np.isnan(values[hist]).any(axis=1) & ~np.isnan(target[hist])]
        if len(ok) < window:
            continue
        ok = ok[-window:]
        if convex:
            lam = CONVEX_PENALTY
            A = np.vstack([values[ok], np.full((1, values.shape[1]), lam)])
            w, _ = nnls(A, np.append(target[ok], lam))
            w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1.0 / len(w))
        else:
            w, _ = nnls(values[ok], target[ok])
        stacked.iloc[i] = values[i] @ w
        weights.iloc[i] = w
    equal = P.mean(axis=1, skipna=False)
    return (equal, stacked, weights) if return_weights else (equal, stacked)


def calibrate_to_mean(
    pred: pd.Series,
    y: pd.Series,
    base: pd.Series,
    window: int | None = None,
    min_obs: int | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Shrink forecasts toward the base by the trailing slope."""
    window = config.CALIBRATION_WINDOW if window is None else window
    min_obs = config.CALIBRATION_MIN if min_obs is None else min_obs
    p, t, b = pred.values, y.values, base.values
    out, slopes = base.copy(), pd.Series(np.nan, index=pred.index)
    for i in range(len(p)):
        if np.isnan(p[i]) or np.isnan(b[i]):
            continue
        hist = np.arange(i)
        ok = hist[~np.isnan(p[hist]) & ~np.isnan(t[hist]) & ~np.isnan(b[hist])][
            -window:
        ]
        slope = 0.0
        if len(ok) >= min_obs:
            d, e = p[ok] - b[ok], t[ok] - b[ok]
            den = float(d @ d)
            slope = float(np.clip((d @ e) / den, 0.0, 1.0)) if den > 0 else 0.0
        slopes.iloc[i] = slope
        out.iloc[i] = b[i] + slope * (p[i] - b[i])
    return out, slopes

import numpy as np
import pandas as pd

from kairos import config
from kairos.models.forecasting import (
    SHARED_FEATURES,
    asset_frame,
    forward_targets,
    historical_mean,
    make_model,
)

POOLED_SPEC = {"pooled_ridge": "ridge", "pooled_xgb": "xgb"}


def long_panel(F, log_price, assets):
    frames = []
    for a in assets:
        rows = asset_frame(F, a)
        own = [c for c in F.columns if c.startswith(a + "_")]
        X = rows[own].copy()
        X.columns = [c[len(a) + 1 :] for c in own]
        for c in SHARED_FEATURES:
            if c in F.columns:
                X[c] = rows[c]
        y = forward_targets(log_price[a].reindex(rows.index))
        X["y"] = y
        X["base"] = historical_mean(y)
        X["pos"] = np.arange(len(rows))
        X["asset"] = a
        frames.append(X)
    panel = pd.concat(frames)
    panel.index.name = "date"
    return panel.reset_index()


def pooled_forecasts(F, log_price, assets, seed=config.SEED, start=None):
    start = config.INITIAL_TRAIN_MONTHS if start is None else start
    panel = long_panel(F, log_price, assets)
    feats = [c for c in panel.columns if c not in ("date", "y", "base", "pos", "asset")]
    X = panel[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    dev = (panel["y"] - panel["base"]).values
    dates = panel["date"].values
    out = {
        a: pd.DataFrame(
            np.nan, index=asset_frame(F, a).index, columns=list(POOLED_SPEC)
        )
        for a in assets
    }
    for d in np.unique(dates[panel["pos"].values >= start]):
        train = (dates < d) & ~np.isnan(dev)
        test = np.where((dates == d) & (panel["pos"].values >= start))[0]
        if train.sum() < 60 or not len(test):
            continue
        for name, kind in POOLED_SPEC.items():
            model = make_model(kind, seed).fit(X[train], dev[train])
            pred = model.predict(X[test])
            for i, p in zip(test, pred):
                a = panel.at[i, "asset"]
                out[a].at[pd.Timestamp(d), name] = panel.at[i, "base"] + p
    return out

import numpy as np
import pandas as pd
import shap
from xgboost import XGBRegressor

import config
from data.loader import month_end_dates
from models.forecasting import asset_frame, feature_columns, forward_targets


def regime_shap(asset, F, log_price, seed=config.SEED):
    rows = asset_frame(F, asset)
    y = forward_targets(log_price.reindex(rows.index)).values
    cols = feature_columns(F, asset)
    X = rows[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    keep = ~np.isnan(y)
    model = XGBRegressor(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=1,
        verbosity=0,
    ).fit(X[keep], y[keep])
    values = np.abs(shap.TreeExplainer(model).shap_values(X[keep]))
    regime = rows["regime"].values[keep]
    frame = pd.DataFrame(values, columns=cols)
    out = pd.DataFrame(
        {
            "overall": frame.mean(),
            "calm": frame[regime == 0].mean(),
            "stressed": frame[regime == 1].mean(),
        }
    ).fillna(0.0)
    out["stressed_minus_calm"] = out["stressed"] - out["calm"]
    return out.sort_values("overall", ascending=False)


def run(paths=config.PATHS, assets=config.SHAP_ASSETS):
    F = pd.read_csv(
        paths.processed / "features_monthly.csv", index_col=0, parse_dates=True
    )
    panel = pd.read_csv(
        paths.processed / "panel_daily.csv", index_col=0, parse_dates=True
    )
    log_price = np.log(panel.reindex(month_end_dates(panel.index)))
    out = {}
    for asset in assets:
        if asset in panel and len(asset_frame(F, asset)) > config.INITIAL_TRAIN_MONTHS:
            out[asset] = regime_shap(asset, F, log_price[asset])
            out[asset].to_csv(paths.tables / f"shap_{asset.replace('-', '_')}.csv")
    return out

"""Per-asset forecasting stage."""

import logging

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from kairos import config
from kairos.data.loader import month_end_dates
from kairos.data.store import load_features, load_panel
from kairos.models.dataset import (
    LSTM_FEATURES,
    asset_frame,
    design_matrix,
    feature_columns,
    forward_targets,
    historical_mean,
)
from kairos.models.ensembles import calibrate_to_mean, stack_forecasts
from kairos.models.pooled import pooled_forecasts
from kairos.models.skill import evaluate, skill_by_period
from kairos.models.walk_forward import walk_forward_lstm, walk_forward_sklearn

log = logging.getLogger(__name__)


def run_asset(
    asset: str,
    F: pd.DataFrame,
    log_price: pd.Series,
    seed: int = config.SEED,
    pooled: pd.DataFrame | None = None,
    initial_train: int | None = None,
) -> pd.DataFrame:
    """All forecasts for one asset."""
    n0 = config.INITIAL_TRAIN_MONTHS if initial_train is None else initial_train
    rows = asset_frame(F, asset)
    y = forward_targets(log_price.reindex(rows.index)).values
    cols = feature_columns(F, asset)
    X = design_matrix(rows, cols)
    preds = walk_forward_sklearn(X, y, ["xgb", "rf", "ridge"], seed, n0)
    lstm_cols = [f"{asset}_{c}" for c in LSTM_FEATURES] + ["VIX_zscore_252d", "regime"]
    X_lstm = design_matrix(rows, lstm_cols)
    preds["lstm"] = walk_forward_lstm(X_lstm, y, seed, n0)
    out = pd.DataFrame({"target": y, **preds}, index=rows.index)
    if pooled is not None:
        for name in config.POOLED:
            out[name] = pooled[name].reindex(out.index)
    out["hist_mean"] = historical_mean(out["target"])
    equal, stacked = stack_forecasts(out[config.MODELS], out["target"])
    out["ensemble_equal"], out["ensemble_stacked"] = equal, stacked
    cand = [c for c in config.CANDIDATES if c in out]
    _, stack_all, weights = stack_forecasts(
        out[cand],
        out["target"],
        fallback=out["hist_mean"],
        return_weights=True,
        convex=True,
    )
    out["stack_all"] = stack_all
    out["ensemble_final"], out["calib_slope"] = calibrate_to_mean(
        stack_all, out["target"], out["hist_mean"]
    )
    for c in cand:
        out[f"w_{c}"] = weights[c]
    gated = ["stack_all", "ensemble_final", "calib_slope"]
    out.loc[out.index[:n0], gated] = np.nan
    return out


def run_all(
    paths: config.Paths = config.PATHS,
    assets: list[str] | None = None,
    n_jobs: int = -1,
    seed: int = config.SEED,
    initial_train: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Forecast every eligible asset and write the CSV files."""
    n0 = config.INITIAL_TRAIN_MONTHS if initial_train is None else initial_train
    F = load_features(paths)
    panel = load_panel(paths)
    log_price = np.log(panel.reindex(month_end_dates(panel.index)))
    chosen = [
        a
        for a in (assets or config.ASSETS)
        if a in panel and len(asset_frame(F, a)) >= n0 + 24
    ]
    log.info("forecasting %d assets, %d initial training months", len(chosen), n0)
    pooled = pooled_forecasts(F, log_price, chosen, seed, n0)
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_asset)(a, F, log_price[a], seed, pooled[a], n0) for a in chosen
    )
    for asset, res in zip(chosen, results):
        res.to_csv(paths.processed / f"forecasts_{asset.replace('-', '_')}.csv")
    return dict(zip(chosen, results))


def load_forecasts(paths: config.Paths = config.PATHS) -> dict[str, pd.DataFrame]:
    """Stored forecasts by asset."""
    out = {}
    for asset in config.ASSETS:
        path = paths.processed / f"forecasts_{asset.replace('-', '_')}.csv"
        if path.exists():
            out[asset] = pd.read_csv(path, index_col=0, parse_dates=True)
    return out


def run(
    paths: config.Paths = config.PATHS,
    assets: list[str] | None = None,
    n_jobs: int = -1,
    initial_train: int | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Forecasting stage."""
    forecasts = run_all(paths, assets, n_jobs, initial_train=initial_train)
    table = evaluate(forecasts)
    table.to_csv(paths.tables / "model_comparison.csv", index=False)
    skill_by_period(forecasts).to_csv(paths.tables / "skill_by_period.csv", index=False)
    weights = pd.DataFrame(
        {
            a: df[[c for c in df.columns if c.startswith("w_")]].mean()
            for a, df in forecasts.items()
        }
    ).T
    weights.columns = [c[2:] for c in weights.columns]
    weights.index.name = "Asset"
    weights.to_csv(paths.tables / "stack_weights.csv")
    return forecasts, table

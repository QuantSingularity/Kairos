"""Forecast diagnostics under alternative designs."""

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr

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
from kairos.models.forecasting import load_forecasts
from kairos.models.skill import r2_oos
from kairos.models.walk_forward import walk_forward_lstm, walk_forward_sklearn

log = logging.getLogger(__name__)

REGIME_FEATURES = ("regime", "p_stressed")
STRESS_FEATURES = REGIME_FEATURES + ("VIX_level", "VIX_chg_5d", "VIX_zscore_252d")


def variants() -> dict[str, tuple[Sequence[str], int]]:
    """Feature sets and initial windows to compare."""
    return {
        "as_specified": ((), config.INITIAL_TRAIN_MONTHS),
        "no_regime_features": (REGIME_FEATURES, config.INITIAL_TRAIN_MONTHS),
        "no_stress_features": (STRESS_FEATURES, config.INITIAL_TRAIN_MONTHS),
        "longer_initial_window": ((), config.DIAG_INITIAL_TRAIN),
        "no_regime_longer_window": (REGIME_FEATURES, config.DIAG_INITIAL_TRAIN),
    }


def variant_forecast(
    asset: str,
    F: pd.DataFrame,
    log_price: pd.Series,
    drop: Sequence[str],
    start: int,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """Walk-forward forecasts for one design."""
    rows = asset_frame(F, asset)
    y = forward_targets(log_price.reindex(rows.index)).values
    cols = [c for c in feature_columns(F, asset) if c not in drop]
    X = design_matrix(rows, cols)
    preds = walk_forward_sklearn(X, y, config.DIAG_MODELS, seed, start)
    return pd.DataFrame({"target": y, **preds}, index=rows.index)


def run_variants(
    F: pd.DataFrame,
    log_price: pd.DataFrame,
    assets: list[str],
    n_jobs: int = -1,
    only: list[str] | None = None,
) -> pd.DataFrame:
    """Forecasts for every design and asset."""
    jobs = [
        (name, a, drop, start)
        for name, (drop, start) in variants().items()
        if name != "as_specified" and (only is None or name in only)
        for a in assets
    ]
    results = Parallel(n_jobs=n_jobs)(
        delayed(variant_forecast)(a, F, log_price[a], drop, start)
        for _, a, drop, start in jobs
    )
    frames = []
    for (name, a, _, _), df in zip(jobs, results):
        frames.append(df.assign(variant=name, asset=a))
    return pd.concat(frames)


def baseline_frames(paths: config.Paths, assets: list[str]) -> pd.DataFrame:
    """Stored forecasts in the diagnostic layout."""
    out = []
    for a, df in load_forecasts(paths).items():
        if a in assets:
            cols = ["target"] + config.DIAG_MODELS
            out.append(df[cols].assign(variant="as_specified", asset=a))
    return pd.concat(out)


def evaluate_variants(frames: pd.DataFrame) -> pd.DataFrame:
    """Skill of each design on common dates."""
    rows = []
    for asset, sub in frames.groupby("asset"):
        wide = {
            v: g.drop(columns=["variant", "asset"]) for v, g in sub.groupby("variant")
        }
        common: pd.Index | None = None
        for g in wide.values():
            ok = g.dropna(subset=["target"] + config.DIAG_MODELS).index
            common = ok if common is None else common.intersection(ok)
        if common is None or len(common) < 12:
            continue
        for v, g in wide.items():
            part = g.loc[common]
            pooled = part[config.DIAG_MODELS].mean(axis=1)
            bench = historical_mean(g["target"]).reindex(common)
            row = {"Asset": asset, "Variant": v, "n_oos": len(common)}
            for m in config.DIAG_MODELS:
                row[f"IC_{m}"] = spearmanr(part[m], part["target"])[0]
            row["IC_mean3"] = spearmanr(pooled, part["target"])[0]
            row["R2OOS_mean3"] = r2_oos(pooled, part["target"], bench)
            rows.append(row)
    return pd.DataFrame(rows)


def summarise_variants(table: pd.DataFrame) -> pd.DataFrame:
    """Mean skill by design."""
    g = table.groupby("Variant")
    out = pd.DataFrame(
        {
            "MeanIC": g["IC_mean3"].mean(),
            "MeanIC_equities": table[table["Asset"].isin(["SPY", "EFA"])]
            .groupby("Variant")["IC_mean3"]
            .mean(),
            "MeanR2OOS": g["R2OOS_mean3"].mean(),
            "AssetsPositiveIC": g["IC_mean3"].apply(lambda s: int((s > 0).sum())),
            "Assets": g["IC_mean3"].count(),
        }
    )
    return out.reindex([v for v in variants() if v in out.index])


def regime_return_link(
    F: pd.DataFrame, log_price: pd.DataFrame, assets: list[str]
) -> pd.DataFrame:
    """Rank correlation of stress probability and next return."""
    rows = []
    cut = config.INITIAL_TRAIN_MONTHS
    for a in assets:
        frame = asset_frame(F, a)
        y = forward_targets(log_price[a].reindex(frame.index))
        d = pd.DataFrame({"p": frame["p_stressed"], "reg": frame["regime"], "y": y})
        d = d.dropna()
        for label, part in (("initial_window", d.iloc[:cut]), ("after", d.iloc[cut:])):
            if len(part) < 12:
                continue
            stressed, calm = part[part["reg"] == 1]["y"], part[part["reg"] == 0]["y"]
            rows.append(
                {
                    "Asset": a,
                    "Window": label,
                    "n": len(part),
                    "CorrStressNextRet": spearmanr(part["p"], part["y"])[0],
                    "MeanRetStressed": (
                        float(stressed.mean()) if len(stressed) else np.nan
                    ),
                    "MeanRetCalm": float(calm.mean()) if len(calm) else np.nan,
                    "StressedMonths": len(stressed),
                }
            )
    return pd.DataFrame(rows)


def lstm_comparison(
    F: pd.DataFrame,
    log_price: pd.DataFrame,
    forecasts: dict[str, pd.DataFrame],
    assets: list[str],
) -> pd.DataFrame:
    """LSTM skill with and without target scaling."""
    rows = []
    for a in assets:
        if a not in forecasts:
            continue
        frame = asset_frame(F, a)
        y = forward_targets(log_price[a].reindex(frame.index))
        cols = [f"{a}_{c}" for c in LSTM_FEATURES] + ["VIX_zscore_252d", "regime"]
        X = design_matrix(frame, cols)
        raw = pd.Series(
            walk_forward_lstm(X, y.values, scale_target=False), index=frame.index
        )
        fixed = forecasts[a]["lstm"]
        d = pd.DataFrame(
            {"y": y, "raw": raw, "fixed": fixed, "hm": historical_mean(y)}
        ).dropna()
        row = {"Asset": a, "n_oos": len(d)}
        for name in ("raw", "fixed"):
            row[f"Disp_{name}"] = float(d[name].std() / d["y"].std())
            row[f"R2OOS_{name}"] = r2_oos(d[name], d["y"], d["hm"])
            row[f"IC_{name}"] = float(spearmanr(d[name], d["y"])[0])
        rows.append(row)
    return pd.DataFrame(rows)


def run(
    paths: config.Paths = config.PATHS,
    assets: list[str] | None = None,
    n_jobs: int = -1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Diagnostics stage."""
    F = load_features(paths)
    panel = load_panel(paths)
    log_price = np.log(panel.reindex(month_end_dates(panel.index)))
    base = baseline_frames(paths, assets or config.ASSETS)
    chosen = sorted(base["asset"].unique())
    log.info("diagnostics for %d assets", len(chosen))
    link = regime_return_link(F, log_price, chosen)
    link.to_csv(paths.tables / "regime_return_link.csv", index=False)
    frames = pd.concat([base, run_variants(F, log_price, chosen, n_jobs)])
    frames.to_csv(paths.processed / "diagnostic_forecasts.csv")
    table = evaluate_variants(frames)
    table.to_csv(paths.tables / "ic_diagnostics.csv", index=False)
    summary = summarise_variants(table)
    summary.to_csv(paths.tables / "ic_diagnostics_summary.csv")
    lstm_comparison(F, log_price, load_forecasts(paths), chosen).to_csv(
        paths.tables / "lstm_repair.csv", index=False
    )
    return link, table, summary

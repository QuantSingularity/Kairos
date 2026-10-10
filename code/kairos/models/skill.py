"""Forecast skill metrics."""

from typing import Any

import numpy as np
import pandas as pd
from kairos import config
from kairos.models.dataset import historical_mean
from scipy.stats import spearmanr


def r2_oos(pred: Any, target: Any, benchmark: Any) -> float:
    """Out-of-sample R2 against a benchmark forecast."""
    denom = float(np.sum((target - benchmark) ** 2))
    return 1.0 - float(np.sum((target - pred) ** 2)) / denom if denom > 0 else np.nan


SKILL_COLUMNS = (
    config.MODELS
    + config.POOLED
    + ["ensemble_equal", "ensemble_stacked", "stack_all", "ensemble_final"]
)


def evaluate(forecasts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Information coefficient, hit rate and R2 by asset and model."""
    rows = []
    for asset, df in forecasts.items():
        cols = [c for c in SKILL_COLUMNS if c in df]
        oos = df.dropna(subset=["target"] + cols)
        row: dict[str, Any] = {"Asset": asset, "n_oos": len(oos)}
        if len(oos) < 12:
            rows.append(row)
            continue
        base = float((oos["target"] > 0).mean())
        bench = historical_mean(df["target"]).reindex(oos.index)
        row["BaseRate"] = base
        row["UpShare_stacked"] = float((oos["ensemble_stacked"] > 0).mean())
        for c in cols:
            ic, p = spearmanr(oos[c], oos["target"])
            row[f"IC_{c}"] = ic
            hit = float((np.sign(oos[c]) == np.sign(oos["target"])).mean())
            row[f"Hit_{c}"] = hit
            row[f"HitEdge_{c}"] = hit - base
            row[f"R2OOS_{c}"] = r2_oos(oos[c], oos["target"], bench)
            row[f"Disp_{c}"] = float(oos[c].std() / oos["target"].std())
            if c == "ensemble_stacked":
                row["p_value_stacked"] = p
            if c == "ensemble_final":
                row["p_value_final"] = p
        singles = [row[f"IC_{m}"] for m in config.MODELS]
        row["MeanSingleIC"] = float(np.mean(singles))
        row["BestSingle"] = config.MODELS[int(np.argmax(singles))]
        row["EnsembleBeatsMeanSingle"] = bool(
            row["IC_ensemble_stacked"] > row["MeanSingleIC"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def skill_by_period(
    forecasts: dict[str, pd.DataFrame], cut: str | None = None
) -> pd.DataFrame:
    """Mean skill before and after the cut date."""
    cut = pd.Timestamp(config.HOLDOUT_START if cut is None else cut)
    periods = (
        ("full", lambda i: np.ones(len(i), dtype=bool)),
        ("development", lambda i: i < cut),
        ("holdout", lambda i: i >= cut),
    )
    rows = []
    for period, pick in periods:
        for c in SKILL_COLUMNS:
            ics, r2s = [], []
            for df in forecasts.values():
                if c not in df:
                    continue
                cols = [k for k in SKILL_COLUMNS if k in df]
                oos = df.dropna(subset=["target"] + cols)
                oos = oos[pick(oos.index)]
                if len(oos) < 12:
                    continue
                bench = historical_mean(df["target"]).reindex(oos.index)
                ics.append(spearmanr(oos[c], oos["target"])[0])
                r2s.append(r2_oos(oos[c], oos["target"], bench))
            if ics:
                rows.append(
                    {
                        "Period": period,
                        "Forecast": c,
                        "MeanIC": float(np.nanmean(ics)),
                        "MeanR2OOS": float(np.nanmean(r2s)),
                        "AssetsPositiveR2": int(np.sum(np.array(r2s) > 0)),
                        "Assets": len(r2s),
                    }
                )
    return pd.DataFrame(rows)

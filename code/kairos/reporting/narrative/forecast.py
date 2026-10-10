"""Narrative paragraphs about the forecasts."""

from typing import Any

import numpy as np
import pandas as pd
from kairos import config


def forecast_paragraph(f: dict[str, Any]) -> str:
    """Summary of forecast skill."""
    text = (
        f"The forecasting layer is the weak link: {f['forecast_summary']} "
        f"Measured against an expanding historical-mean forecast, the original stacked ensemble has a negative out-of-sample R2 on "
        f"{f['r2_neg_assets']} of {f['n_assets']} assets (mean {f['mean_r2']:.3f}), and across all models and assets "
        f"{f['r2_neg_pairs']} of {f['r2_pairs']} model-asset R2 values are negative."
    )
    spy = f["spy"]
    if spy is not None:
        text += (
            f" Hit rates flatter the models: for SPY the original stacked ensemble calls an up month in {spy['UpShare_stacked']:.0%} of months "
            f"and is right {spy['Hit_ensemble_stacked']:.1%} of the time, while SPY rose in {spy['BaseRate']:.1%} of months"
            + (
                ", so an always-up rule would have scored higher"
                if spy["Hit_ensemble_stacked"] < spy["BaseRate"]
                else ""
            )
            + f". The stacked hit rate is below the always-up baseline on {f['hit_below_base']} of {f['n_assets']} assets."
        )
    if f["lstm_disp"] > 1.5:
        text += (
            f" The LSTM is badly scaled: its forecasts vary about {f['lstm_disp']:.1f} times as much as realised returns, "
            f"with a mean R2 of {f['lstm_r2']:.1f}, which injects noise into the equal-weight ensemble."
        )
    return text


def diagnosis_paragraph(res: dict[str, Any]) -> str:
    """Explanation of the information coefficient diagnostics."""
    link, summ = res["regime_link"], res["ic_summary"]
    parts = []
    spy = link[link["Asset"] == "SPY"].set_index("Window")
    if {"initial_window", "after"} <= set(spy.index):
        c1 = spy.at["initial_window", "CorrStressNextRet"]
        c2 = spy.at["after", "CorrStressNextRet"]
        both = link.pivot(
            index="Asset", columns="Window", values="CorrStressNextRet"
        ).dropna()
        flips = int((both["initial_window"] * both["after"] < 0).sum())
        text = (
            f"One candidate explanation is that the first {config.INITIAL_TRAIN_MONTHS} months of training are dominated by 2008. "
            f"For SPY, the rank correlation between the filtered stress probability and the next-month return is {c1:+.2f} in the "
            f"initial training window and {c2:+.2f} afterwards, and the sign differs between the two windows for {flips} of "
            f"{len(both)} assets."
        )
        text += (
            " For SPY this is consistent with models learning a stress-precedes-losses rule from the crisis that the "
            "rebounds of later years then contradicted."
            if c1 * c2 < 0
            else " For SPY the sign does not change, so a reversed stress relationship does not by itself explain the result."
        )
        parts.append(text)
    if "as_specified" in summ.index and len(summ) > 1:
        base = summ.at["as_specified", "MeanIC"]
        others = summ.drop("as_specified")["MeanIC"]
        best = others.idxmax()
        desc = "; ".join(f"{n.replace('_', ' ')} {v:+.3f}" for n, v in others.items())
        text = (
            f"Re-fitting the tree and linear models under {len(others)} alternative designs on the dates common to all designs gives a "
            f"mean IC across assets of {base:+.3f} as specified, then {desc}."
        )
        gain = others[best] - base
        text += (
            f" The largest improvement comes from the {best.replace('_', ' ')} design ({gain:+.3f})."
            if gain > 0
            else " No alternative design improves the mean IC."
        )
        if (summ["MeanR2OOS"] < 0).all():
            text += (
                " The R2 against the historical mean stays negative under every design, so no design yields a forecast that "
                "beats the historical mean, and the weak skill is not simply an artefact of the regime features or the initial "
                "window. It points to a weak and unstable one-month return signal."
            )
        else:
            text += " At least one design has a positive mean R2 against the historical mean."
        if gain > 0:
            text += (
                " Choosing a design after seeing out-of-sample results is a post hoc choice, so the pipeline keeps the specified "
                "design and treats this only as a diagnostic that needs fresh data to confirm."
            )
        parts.append(text)
    return " ".join(parts)


def skill_value(skill: pd.DataFrame, forecast: str, period: str) -> float:
    """Mean R2 for one forecast and period."""
    row = skill[(skill["Forecast"] == forecast) & (skill["Period"] == period)]
    return float(row["MeanR2OOS"].iloc[0]) if len(row) else np.nan


def repair_paragraph(res: dict[str, Any]) -> str:
    """Effect of the forecast repairs."""
    lr, skill = res["lstm_repair"], res["skill_period"]
    if lr.empty or skill.empty:
        return ""
    cut = pd.Timestamp(config.HOLDOUT_START).date()
    text = (
        f"The LSTM trained on raw monthly returns produced forecasts about {lr['Disp_raw'].mean():.1f} times as variable as "
        f"realised returns (mean R2 {lr['R2OOS_raw'].mean():.1f}). Scaling the training target by its training mean and standard "
        f"deviation brings the ratio to {lr['Disp_fixed'].mean():.2f} and the mean R2 to {lr['R2OOS_fixed'].mean():.2f}. "
        f"Two pooled cross-asset models, a convex stacker that includes the historical mean as a candidate, and a causal calibration "
        f"that shrinks the stacked forecast toward the historical mean by the trailing {config.CALIBRATION_WINDOW}-month "
        "slope were added."
    )
    final = {
        p: skill_value(skill, config.VIEW_COLUMN, p)
        for p in ("full", "development", "holdout")
    }
    raw = {
        p: skill_value(skill, config.RAW_VIEW_COLUMN, p)
        for p in ("full", "development", "holdout")
    }
    text += (
        f" Mean R2 against the historical mean is {final['full']:.3f} for the calibrated forecast, against {raw['full']:.3f} for "
        f"the original stacked ensemble. Before {cut} the values are {final['development']:.3f} and {raw['development']:.3f}; "
        f"from {cut} they are {final['holdout']:.3f} and {raw['holdout']:.3f}. IC is not informative for a forecast that is mostly the "
        "historical mean, so R2 is the metric used here."
    )
    positive = skill[(skill["Period"] != "full") & (skill["MeanR2OOS"] > 0)]
    if positive.empty:
        text += (
            " No forecast, single model or ensemble, has a positive mean R2 in either period, so the repairs remove the harm done by "
            "badly scaled forecasts but do not create skill. The calibrated forecast behaves almost like the historical mean, "
            "which remains the best return forecast in this sample."
        )
    else:
        names = ", ".join(sorted(set(positive["Forecast"])))
        text += f" Forecasts with a positive mean R2 in at least one period: {names}."
    return text

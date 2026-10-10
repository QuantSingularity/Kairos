"""Discussion and summary text assembled from the paragraphs."""

from typing import Any

from kairos import config
from kairos.reporting.narrative.forecast import (
    diagnosis_paragraph,
    forecast_paragraph,
    repair_paragraph,
    skill_value,
)
from kairos.reporting.narrative.portfolio import (
    exploratory_paragraph,
    overlay_paragraph,
    sample_paragraph,
    sensitivity_paragraph,
    stress_paragraph,
)
from kairos.reporting.narrative.robustness import (
    holdout_paragraph,
    period_paragraph,
    prior_paragraph,
    window_paragraph,
)


def discussion(res: dict[str, Any], f: dict[str, Any]) -> list[str]:
    """Discussion section paragraphs."""
    parts = [
        f["answer"],
        forecast_paragraph(f),
        repair_paragraph(res),
        diagnosis_paragraph(res),
        overlay_paragraph(res),
        stress_paragraph(res),
        sample_paragraph(res),
        period_paragraph(res),
        window_paragraph(res),
        exploratory_paragraph(res),
        sensitivity_paragraph(res),
        prior_paragraph(res),
        holdout_paragraph(res),
    ]
    return [p for p in parts if p]


def summary_points(res: dict[str, Any], f: dict[str, Any]) -> list[str]:
    """Summary of findings paragraphs."""
    s = res["summary"]
    points = [
        f["answer"],
        f"Regime ablation: the Sharpe difference between BL_ML_Regime and BL_ML_NoRegime is {f['ablation']}. {f['stress_cvar']}",
        "Forecasting: " + f["forecast_summary"],
    ]
    if {"Regime_60_40", "Static_60_40"} <= set(s.index):
        points.append(
            f"Overlay ablation: the same stress rule applied to static 60/40 gives a Sharpe ratio of {s.at['Regime_60_40', 'Sharpe']:.2f} "
            f"against {s.at['Static_60_40', 'Sharpe']:.2f}, with a maximum drawdown of {s.at['Regime_60_40', 'MaxDrawdown']:.1%} "
            f"against {s.at['Static_60_40', 'MaxDrawdown']:.1%}."
        )
    link = (
        res["regime_link"]
        .pivot(index="Asset", columns="Window", values="CorrStressNextRet")
        .dropna()
    )
    flips = int((link["initial_window"] * link["after"] < 0).sum())
    summ = res["ic_summary"]
    points.append(
        f"Diagnosis: the stress-to-return relationship changes sign between the initial training window and later years for "
        f"{flips} of {len(link)} assets, and the R2 against the historical mean is "
        + (
            "negative under every re-fitted design."
            if (summ["MeanR2OOS"] < 0).all()
            else "positive under at least one re-fitted design."
        )
    )
    t = res["sensitivity"]
    points.append(
        f"Robustness: {int((t['Sharpe'] > t['Sharpe_60_40']).sum())} of {len(t)} Black-Litterman parameter settings exceed the "
        f"Sharpe ratio of static 60/40."
    )
    points.append(
        f"Regimes: the walk-forward HMM flags {f['hmm_stress']:.0%} of days inside the documented stress windows and "
        f"{f['hmm_calm']:.0%} of days inside the calm control windows, against {f['vol_stress']:.0%} and {f['vol_calm']:.0%} for the volatility rule."
    )
    skill = res["skill_period"]
    pos = skill[(skill["Period"] != "full") & (skill["MeanR2OOS"] > 0)]
    points.append(
        f"Forecast repair: the LSTM's dispersion falls from {res['lstm_repair']['Disp_raw'].mean():.1f}x to "
        f"{res['lstm_repair']['Disp_fixed'].mean():.2f}x realised returns, and the calibrated forecast has a mean R2 of "
        f"{skill_value(skill, config.VIEW_COLUMN, 'full'):.3f} against the historical mean. "
        + (
            "No forecast has a positive mean R2 in either the development or the holdout period."
            if pos.empty
            else "Some forecasts have a positive mean R2 in at least one period."
        )
    )
    pr = res["prior"][res["prior"]["Variant"] == "BL_ML_Regime"]
    points.append(
        "Prior: BL_ML_Regime adds a significant Sharpe gain over its own prior portfolio for "
        f"{int(((pr['p_value'] < 0.05) & (pr['Diff'] > 0)).sum())} of {len(pr)} priors tested."
    )
    w = res["window"]
    if "Regime_60_40" in w.index:
        points.append(
            f"Crisis window: Regime_60_40 has a maximum drawdown of {w.at['Regime_60_40', 'MaxDD_crisis_window']:.1%} against "
            f"{w.at['Static_60_40', 'MaxDD_crisis_window']:.1%} for static 60/40 when 2008 is included."
        )
    points.append(holdout_paragraph(res))
    points.append("Sharpe ranking: " + f["ranking"] + ".")
    return points

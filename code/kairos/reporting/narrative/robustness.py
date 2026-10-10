"""Narrative paragraphs about robustness checks."""

from typing import Any

import pandas as pd

from kairos import config
from kairos.reporting.results import describe_diff, pair


def window_paragraph(res: dict[str, Any]) -> str:
    """Crisis-inclusive window."""
    w, meta, sig = res["window"], res["window_meta"], res["window_sig"]
    need = ["BL_ML_Regime", "Static_60_40", "Regime_60_40"]
    if w.empty or not all(n in w.index for n in need):
        return ""
    start = meta["first_decision"].iloc[0]
    months = int(w["Months_crisis_window"].iloc[0])
    text = (
        f"A crisis-inclusive window starts the out-of-sample forecasts in {start} with a {int(meta['initial_train_months'].iloc[0])}-month "
        f"initial training period, which adds the 2008 crisis to the test ({months} months). In that window Static_60_40 has a "
        f"Sharpe ratio of {w.at['Static_60_40', 'Sharpe_crisis_window']:.2f} and a maximum drawdown of "
        f"{w.at['Static_60_40', 'MaxDD_crisis_window']:.1%}, Regime_60_40 has {w.at['Regime_60_40', 'Sharpe_crisis_window']:.2f} and "
        f"{w.at['Regime_60_40', 'MaxDD_crisis_window']:.1%}, and BL_ML_Regime has {w.at['BL_ML_Regime', 'Sharpe_crisis_window']:.2f} "
        f"and {w.at['BL_ML_Regime', 'MaxDD_crisis_window']:.1%}."
    )
    row = pair(sig, "Regime_60_40", "Static_60_40")
    if row is not None:
        text += f" The Sharpe difference between the overlay and plain 60/40 is {describe_diff(row)}."
    row = pair(sig, "BL_ML_Regime", "Regime_60_40")
    if row is not None:
        text += f" BL_ML_Regime against the overlay: {describe_diff(row)}."
    return text


def period_paragraph(res: dict[str, Any]) -> str:
    """Development and holdout periods."""
    t, sig = res["period"], res["period_sig"]
    names = [
        n
        for n in ("BL_ML_Regime", "Static_60_40", "Regime_60_40")
        if (n, "holdout") in t.index
    ]
    if not names:
        return ""
    cut = pd.Timestamp(config.HOLDOUT_START).date()
    cells = "; ".join(
        f"{n} {t.at[(n, 'development'), 'Sharpe']:.2f} before and {t.at[(n, 'holdout'), 'Sharpe']:.2f} from {cut}"
        for n in names
    )
    text = f"Splitting the sample at {cut} gives Sharpe ratios of {cells}."
    row = sig[
        (sig["period"] == "holdout")
        & (sig["strategy"] == "BL_ML_Regime")
        & (sig["benchmark"] == "Static_60_40")
    ]
    if len(row):
        text += f" In the later period the Sharpe difference between BL_ML_Regime and static 60/40 is {describe_diff(row.iloc[0])}."
    text += (
        " This split is a reporting device, not a clean test: the repairs were motivated by full-sample diagnostics, so only months "
        "that arrive after the freeze date are untouched."
    )
    return text


def prior_paragraph(res: dict[str, Any]) -> str:
    """Prior comparison."""
    t = res["prior"]
    if t.empty:
        return ""
    sub = t[t["Variant"] == "BL_ML_Regime"]
    cells = "; ".join(
        f"{r.Prior} {r.Diff:+.2f} (p = {r.p_value:.2f})" for r in sub.itertuples()
    )
    text = (
        "The Sharpe ratio of a Black-Litterman portfolio depends on the prior, because each prior is a different starting portfolio, "
        "so the right comparison is against that prior's own portfolio. The Sharpe difference of BL_ML_Regime to its prior portfolio is: "
        + cells
        + "."
    )
    sig_any = bool(((sub["p_value"] < 0.05) & (sub["Diff"] > 0)).any())
    text += (
        " At least one prior shows a significant gain over its own portfolio."
        if sig_any
        else " No prior shows a significant gain, so the Black-Litterman layer adds little beyond the prior it starts from."
    )
    if "observed" in set(t["Prior"]):
        text += " An observed prior supplied in data/raw is included."
    else:
        text += (
            " No observed market-capitalisation series is bundled, because none could be sourced for these exchange-traded funds "
            "without hand-picking splits; the pipeline accepts one in data/raw/prior_weights.csv and then adds it to this comparison."
        )
    return text


def holdout_paragraph(res: dict[str, Any]) -> str:
    """Frozen specification and new months."""
    st_ = res["holdout_status"].iloc[0]
    text = (
        f"The specification is frozen as of {st_['freeze_date']} (hash {st_['spec_hash']}). "
        f"{int(st_['months_since_freeze'])} realised months have been added since then"
    )
    if not st_["spec_unchanged"]:
        return (
            text
            + ", but the code or parameters have changed since the freeze, so they are not confirmatory evidence."
        )
    if int(st_["months_since_freeze"]) == 0:
        return (
            text
            + ", so there is no confirmatory evidence yet. Re-run with refreshed data to evaluate months as they arrive."
        )
    return text + (
        ", enough for inference."
        if st_["inference_possible"]
        else ", too few for inference (at least 24 are needed)."
    )

"""Narrative paragraphs about the portfolios."""

from typing import Any

from kairos import config
from kairos.reporting.findings import KAIROS_MAIN
from kairos.reporting.results import describe_diff, pair, require_pair


def overlay_paragraph(res: dict[str, Any]) -> str:
    """Forecasts against mechanical overlays."""
    s, sig = res["summary"], res["significance"]
    need = ["Regime_60_40", "Regime_Market", "Static_60_40", "Market_Reference"]
    if not all(n in s.index for n in need):
        return ""
    mult = config.STRESSED_DELTA_MULT
    d60 = require_pair(sig, "Regime_60_40", "Static_60_40")
    dmk = require_pair(sig, "Regime_Market", "Market_Reference")
    text = (
        f"The decisive ablation applies a mechanical version of the same stress rule (risky weights divided by {mult:g}, "
        f"the remainder in cash) to benchmarks that use no forecasts. On static 60/40 it moves the Sharpe ratio from "
        f"{s.at['Static_60_40', 'Sharpe']:.2f} to {s.at['Regime_60_40', 'Sharpe']:.2f} (difference {d60['diff']:+.3f}, p = {d60['p_value']:.3f}) "
        f"and the maximum drawdown from {s.at['Static_60_40', 'MaxDrawdown']:.1%} to {s.at['Regime_60_40', 'MaxDrawdown']:.1%}. "
        f"On the market reference the Sharpe ratio moves from {s.at['Market_Reference', 'Sharpe']:.2f} to "
        f"{s.at['Regime_Market', 'Sharpe']:.2f} (p = {dmk['p_value']:.3f}) and the drawdown from "
        f"{s.at['Market_Reference', 'MaxDrawdown']:.1%} to {s.at['Regime_Market', 'MaxDrawdown']:.1%}."
    )
    best_overlay = s.loc[list(config.OVERLAYS), "Sharpe"].idxmax()
    beats = [n for n in KAIROS_MAIN if s.at[n, "Sharpe"] > s.at[best_overlay, "Sharpe"]]
    text += (
        f" The best overlay ({best_overlay}) has a Sharpe ratio of {s.at[best_overlay, 'Sharpe']:.2f}, against "
        f"{s.at['BL_ML_Regime', 'Sharpe']:.2f} for BL_ML_Regime and {s.at['BL_ML_Regime_IC', 'Sharpe']:.2f} for BL_ML_Regime_IC."
    )
    p1 = pair(sig, "BL_ML_Regime", best_overlay)
    p2 = pair(sig, "BL_ML_Regime_IC", best_overlay)
    tests = ""
    if p1 is not None and p2 is not None:
        tests = (
            f" (paired Sharpe differences {p1['diff']:+.2f} and {p2['diff']:+.2f}, "
            f"p = {p1['p_value']:.2f} and {p2['p_value']:.2f})"
        )
    text += (
        " The forecast-driven portfolios exceed the overlay"
        + tests
        + ", but the difference is not shown to be real."
        if beats
        else " Neither forecast-driven portfolio exceeds a rule that needs no forecasts"
        + tests
        + ", so the machine learning and Black-Litterman machinery shows no measurable benefit beyond the defensive exposure cut."
    )
    if config.RAW in s.index:
        n_raw = pair(sig, "BL_ML_Regime", config.RAW)
        text += (
            f" With the original, uncalibrated views the Sharpe ratio is {s.at[config.RAW, 'Sharpe']:.2f}, and the Sharpe difference "
            f"of the calibrated BL_ML_Regime to it is {describe_diff(n_raw)}."
        )
    if config.NAIVE in s.index:
        n_ml = require_pair(sig, "BL_ML_Regime", config.NAIVE)
        text += (
            f" Black-Litterman with naive historical-mean views scores {s.at[config.NAIVE, 'Sharpe']:.2f}, and the Sharpe "
            f"difference of BL_ML_Regime to it is {describe_diff(n_ml)}"
            + (
                ", so the machine learning views do not improve on naive ones inside the same allocator."
                if n_ml["diff"] <= 0
                else "."
            )
        )
    return text


def stress_paragraph(res: dict[str, Any]) -> str:
    """Behaviour in stressed months."""
    rm = res["regime_metrics"].xs("Stressed", level=1)
    s = res["summary"]
    names = ["BL_ML_Regime", "Market_Reference", "Equal_Weight"]
    if not all(n in rm.index for n in names):
        return ""
    k, mk, ew = (rm.at[n, "AnnReturn"] for n in names)
    text = (
        f"The regime layer does reduce risk. BL_ML_Regime has a maximum drawdown of {s.at['BL_ML_Regime', 'MaxDrawdown']:.1%} "
        f"against {s.at['BL_ML_NoRegime', 'MaxDrawdown']:.1%} without regime scaling, although the Sharpe gain is not significant. "
        f"The cost shows in months flagged stressed at the decision date: BL_ML_Regime earned {k:.1%} annualised in those months, "
        f"against {mk:.1%} for Market_Reference and {ew:.1%} for Equal_Weight."
    )
    if k < min(mk, ew):
        text += (
            " Stress flags in this sample were typically followed by rebounds, so cutting exposure protects capital in the "
            "drawdown but gives up return in the recovery."
        )
    return text


def sample_paragraph(res: dict[str, Any]) -> str:
    """Statistical power of the sample."""
    sub = res["subperiod"]
    if "Static_60_40" not in sub.index:
        return ""
    a = sub.at["Static_60_40", "FirstHalfSharpe"]
    b = sub.at["Static_60_40", "SecondHalfSharpe"]
    fell = int((sub["SecondHalfSharpe"] < sub["FirstHalfSharpe"]).sum())
    return (
        f"The main backtest starts in January 2009, after the crisis trough, so it covers a long equity bull market that favours "
        f"benchmarks with a high equity weight and leaves the 2008 drawdown outside the main window; the crisis-inclusive window "
        f"below adds it. The Sharpe ratio of static 60/40 is "
        f"{a:.2f} in the first half of the sample and {b:.2f} in the second, and {fell} of {len(sub)} strategies show a lower "
        f"Sharpe ratio in the second half, so conclusions about relative performance depend on the period."
    )


def exploratory_paragraph(res: dict[str, Any]) -> str:
    """Exploratory comparisons."""
    s, d, sig = res["summary"], res["deflated"], res["significance"]
    kair = [n for n in config.KAIROS if n in s.index]
    best = s.loc[kair, "Sharpe"].idxmax()
    lo, hi = config.DSR_TRIALS
    dsr = {t: d[(d["strategy"] == best) & (d["Trials"] == t)] for t in (lo, hi)}
    if any(v.empty for v in dsr.values()):
        return ""
    text = (
        f"{best} has the highest Sharpe ratio of the three Kairos variants, but it is the best of several variants tried on one "
        f"sample (five counting the raw-view and historical-mean ablations) and should be read as exploratory. Its deflated Sharpe ratio gives a probability of {dsr[lo].iloc[0]['DSR']:.3f} "
        f"that the true Sharpe ratio is positive after allowing for {lo} trials and {dsr[hi].iloc[0]['DSR']:.3f} after allowing for {hi}. "
        "This tests the Sharpe ratio against zero, not against 60/40, so it does not contradict the paired comparisons above."
    )
    others = [n for n in KAIROS_MAIN if n != best]
    if best in KAIROS_MAIN and others:
        gap = pair(sig, best, others[0])
        if gap is not None:
            text += f" Against its sibling {others[0]} the Sharpe difference is {describe_diff(gap)}."
    return text


def sensitivity_paragraph(res: dict[str, Any]) -> str:
    """Parameter sensitivity."""
    t = res["sensitivity"]
    if t.empty:
        return ""
    ref = float(t["Sharpe_60_40"].iloc[0])
    n_above, n = int((t["Sharpe"] > ref).sum()), len(t)
    ranges = []
    for v in KAIROS_MAIN:
        sub = t[t["Variant"] == v]
        ranges.append(
            f"{v} from {sub['Sharpe'].min():.2f} to {sub['Sharpe'].max():.2f}"
        )
    params = ", ".join(t["Parameter"].unique())
    text = (
        f"Across {n} one-at-a-time parameter changes ({params}), the Sharpe ratio ranges "
        + " and ".join(ranges)
        + f", against {ref:.2f} for Static_60_40, and {n_above} of {n} runs exceed the 60/40 value."
    )
    if n_above == 0:
        text += " No setting in the grid closes the gap, so the shortfall is not a tuning artefact."
    elif n_above < n / 2:
        text += " Only a minority of settings beat 60/40, so any apparent edge depends on tuning."
    else:
        text += " Most settings beat 60/40, so the shortfall is specific to the default parameters."
    eq = t[(t["Parameter"] == "prior") & (t["Value"] == "equal")]
    if len(eq) == len(KAIROS_MAIN):
        vals = " and ".join(f"{r.Sharpe:.2f}" for r in eq.itertuples())
        text += (
            f" An equal-weight prior in place of the reference allocation gives {vals}."
        )
    return text

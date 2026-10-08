import numpy as np
import pandas as pd

import config

KAIROS_MAIN = ("BL_ML_Regime", "BL_ML_Regime_IC")


def load_results(paths=config.PATHS):
    t = paths.tables
    return {
        "quality": pd.read_csv(paths.processed / "data_quality.csv"),
        "regime_validation": pd.read_csv(t / "regime_validation.csv"),
        "models": pd.read_csv(t / "model_comparison.csv"),
        "summary": pd.read_csv(t / "summary_metrics.csv", index_col=0),
        "regime_metrics": pd.read_csv(t / "regime_metrics.csv", index_col=[0, 1]),
        "cost": pd.read_csv(t / "cost_sensitivity.csv", index_col=0),
        "subperiod": pd.read_csv(t / "subperiod_sharpe.csv", index_col=0),
        "significance": pd.read_csv(t / "significance.csv"),
        "deflated": pd.read_csv(t / "deflated_sharpe.csv"),
        "sensitivity": pd.read_csv(t / "sensitivity.csv"),
        "ic_diag": pd.read_csv(t / "ic_diagnostics.csv"),
        "ic_summary": pd.read_csv(t / "ic_diagnostics_summary.csv", index_col=0),
        "regime_link": pd.read_csv(t / "regime_return_link.csv"),
    }


def md_table(df, formats=None, index=True):
    formats = formats or {}
    frame = df.reset_index() if index else df
    head = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    lines = [head, rule]
    for _, row in frame.iterrows():
        cells = []
        for c in frame.columns:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append(
                    "n/a" if np.isnan(v) else formats.get(c, "{:.3f}").format(v)
                )
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def describe_diff(row):
    verdict = (
        "statistically significant at the 5% level"
        if row["p_value"] < 0.05
        else "not statistically significant"
    )
    return (
        f"{row['diff']:+.2f} (95% interval {row['ci_low']:.2f} to {row['ci_high']:.2f}, "
        f"p = {row['p_value']:.3f}), {verdict}"
    )


def pair(sig, a, b):
    sub = sig[(sig["strategy"] == a) & (sig["benchmark"] == b)]
    return sub.iloc[0] if len(sub) else None


def _skill_counts(m):
    cols = config.MODELS + ["ensemble_equal", "ensemble_stacked"]
    r2 = m[[f"R2OOS_{c}" for c in cols if f"R2OOS_{c}" in m]].to_numpy().ravel()
    r2 = r2[~np.isnan(r2)]
    return int((r2 < 0).sum()), len(r2)


def findings(res):
    s, sig = res["summary"], res["significance"]
    out = {}
    ranked = s["Sharpe"].sort_values(ascending=False)
    out["ranking"] = ", ".join(f"{n} ({v:.2f})" for n, v in ranked.items())
    core = [n for n in list(config.KAIROS) + list(config.CLASSICAL) if n in s.index]
    top = s.loc[core, "Sharpe"].idxmax()
    classical = [n for n in config.CLASSICAL if n in s.index]
    best_classical = s.loc[classical, "Sharpe"].idxmax()
    out["top"], out["best_classical"] = top, best_classical
    wins = False
    for name in KAIROS_MAIN:
        sub = sig[sig["strategy"] == name]
        vs_best = pair(sig, name, best_classical)
        out[f"{name}_vs_best"] = describe_diff(vs_best)
        wins = wins or (vs_best["diff"] > 0 and vs_best["p_value"] < 0.05)
        rows = sub[sub["benchmark"].isin(classical)]
        out[f"{name}_better"] = int(
            ((rows["diff"] > 0) & (rows["p_value"] < 0.05)).sum()
        )
        out[f"{name}_worse"] = int(
            ((rows["diff"] < 0) & (rows["p_value"] < 0.05)).sum()
        )
        out[f"{name}_n"] = len(rows)
    out["supported"] = bool(wins)
    out["ablation"] = describe_diff(pair(sig, "BL_ML_Regime", "BL_ML_NoRegime"))
    stressed = res["regime_metrics"].xs("Stressed", level=1)
    out["stress_cvar"] = (
        f"In stressed months the regime-aware portfolio's worst-5% average monthly return was "
        f"{stressed.loc['BL_ML_Regime', 'CVaR5']:.2%} against {stressed.loc['BL_ML_NoRegime', 'CVaR5']:.2%} without regime scaling."
    )
    m = res["models"].dropna(subset=["IC_ensemble_stacked"])
    out["n_assets"] = len(m)
    out["mean_ic"] = float(m["IC_ensemble_stacked"].mean())
    pos = m[(m["p_value_stacked"] < 0.05) & (m["IC_ensemble_stacked"] > 0)]
    neg = m[(m["p_value_stacked"] < 0.05) & (m["IC_ensemble_stacked"] < 0)]
    out["sig_pos"], out["sig_neg"] = len(pos), len(neg)
    out["neg_assets"] = ", ".join(neg["Asset"])
    out["pos_assets"] = ", ".join(pos["Asset"])
    out["ens_beats"] = int(m["EnsembleBeatsMeanSingle"].sum())
    out["r2_neg_assets"] = int((m["R2OOS_ensemble_stacked"] < 0).sum())
    out["mean_r2"] = float(m["R2OOS_ensemble_stacked"].mean())
    out["hit_below_base"] = int((m["HitEdge_ensemble_stacked"] < 0).sum())
    out["r2_neg_pairs"], out["r2_pairs"] = _skill_counts(m)
    out["lstm_disp"] = float(m["Disp_lstm"].mean())
    out["lstm_r2"] = float(m["R2OOS_lstm"].mean())
    out["spy"] = m[m["Asset"] == "SPY"].iloc[0] if (m["Asset"] == "SPY").any() else None
    rv = res["regime_validation"]
    out["hmm_stress"] = float(rv.loc[rv["Kind"] == "stress", "Rate_hmm"].mean())
    out["hmm_calm"] = float(rv.loc[rv["Kind"] == "calm", "Rate_hmm"].mean())
    out["vol_stress"] = float(rv.loc[rv["Kind"] == "stress", "Rate_vol"].mean())
    out["vol_calm"] = float(rv.loc[rv["Kind"] == "calm", "Rate_vol"].mean())
    out["answer"] = (
        f"The research hypothesis is {'supported' if wins else 'not supported'}. The highest Sharpe ratio among the Kairos "
        f"portfolios and the classical benchmarks belongs to {top}. The Sharpe difference of BL_ML_Regime to the best classical "
        f"benchmark ({best_classical}) is {out['BL_ML_Regime_vs_best']}; BL_ML_Regime is significantly better than "
        f"{out['BL_ML_Regime_better']} and significantly worse than {out['BL_ML_Regime_worse']} of the "
        f"{out['BL_ML_Regime_n']} classical benchmarks (5% level). For BL_ML_Regime_IC the difference is "
        f"{out['BL_ML_Regime_IC_vs_best']}."
    )
    out["forecast_summary"] = (
        f"the stacked ensemble has a mean out-of-sample information coefficient of {out['mean_ic']:.3f}. "
        f"{out['sig_neg']} of {out['n_assets']} assets show an IC that is significantly negative at the 5% level"
        + (f" ({out['neg_assets']})" if out["sig_neg"] else "")
        + f" and {out['sig_pos']} show one that is significantly positive"
        + (f" ({out['pos_assets']})" if out["sig_pos"] else "")
        + f". The stacked ensemble beats the mean of the single models on {out['ens_beats']} of {out['n_assets']} assets."
    )
    return out


def _forecast_paragraph(f):
    text = (
        f"The forecasting layer is the weak link: {f['forecast_summary']} "
        f"Measured against an expanding historical-mean forecast, the stacked ensemble has a negative out-of-sample R2 on "
        f"{f['r2_neg_assets']} of {f['n_assets']} assets (mean {f['mean_r2']:.3f}), and across all models and assets "
        f"{f['r2_neg_pairs']} of {f['r2_pairs']} model-asset R2 values are negative."
    )
    spy = f["spy"]
    if spy is not None:
        text += (
            f" Hit rates flatter the models: for SPY the stacked ensemble calls an up month in {spy['UpShare_stacked']:.0%} of months "
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


def _diagnosis_paragraph(res):
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


def _overlay_paragraph(res):
    s, sig = res["summary"], res["significance"]
    need = ["Regime_60_40", "Regime_Market", "Static_60_40", "Market_Reference"]
    if not all(n in s.index for n in need):
        return ""
    mult = config.STRESSED_DELTA_MULT
    d60 = pair(sig, "Regime_60_40", "Static_60_40")
    dmk = pair(sig, "Regime_Market", "Market_Reference")
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
    if config.NAIVE in s.index:
        n_ml = pair(sig, "BL_ML_Regime", config.NAIVE)
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


def _stress_paragraph(res):
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


def _sample_paragraph(res):
    sub = res["subperiod"]
    if "Static_60_40" not in sub.index:
        return ""
    a = sub.at["Static_60_40", "FirstHalfSharpe"]
    b = sub.at["Static_60_40", "SecondHalfSharpe"]
    fell = int((sub["SecondHalfSharpe"] < sub["FirstHalfSharpe"]).sum())
    return (
        f"The backtest starts in January 2009, after the crisis trough, so it covers a long equity bull market that favours "
        f"benchmarks with a high equity weight and leaves the 2008 drawdown outside the test. The Sharpe ratio of static 60/40 is "
        f"{a:.2f} in the first half of the sample and {b:.2f} in the second, and {fell} of {len(sub)} strategies show a lower "
        f"Sharpe ratio in the second half, so conclusions about relative performance depend on the period."
    )


def _exploratory_paragraph(res):
    s, d, sig = res["summary"], res["deflated"], res["significance"]
    kair = [n for n in config.KAIROS if n in s.index]
    best = s.loc[kair, "Sharpe"].idxmax()
    lo, hi = config.DSR_TRIALS
    dsr = {t: d[(d["strategy"] == best) & (d["Trials"] == t)] for t in (lo, hi)}
    if any(v.empty for v in dsr.values()):
        return ""
    text = (
        f"{best} has the highest Sharpe ratio of the three Kairos variants, but it is the best of {len(kair)} variants tried on one "
        f"sample and should be read as exploratory. Its deflated Sharpe ratio gives a probability of {dsr[lo].iloc[0]['DSR']:.3f} "
        f"that the true Sharpe ratio is positive after allowing for {lo} trials and {dsr[hi].iloc[0]['DSR']:.3f} after allowing for {hi}. "
        "This tests the Sharpe ratio against zero, not against 60/40, so it does not contradict the paired comparisons above."
    )
    others = [n for n in KAIROS_MAIN if n != best]
    if best in KAIROS_MAIN and others:
        gap = pair(sig, best, others[0])
        if gap is not None:
            text += f" Against its sibling {others[0]} the Sharpe difference is {describe_diff(gap)}."
    return text


def _sensitivity_paragraph(res):
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


def discussion(res, f):
    parts = [
        f["answer"],
        _forecast_paragraph(f),
        _diagnosis_paragraph(res),
        _overlay_paragraph(res),
        _stress_paragraph(res),
        _sample_paragraph(res),
        _exploratory_paragraph(res),
        _sensitivity_paragraph(res),
    ]
    return [p for p in parts if p]


def summary_points(res, f):
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
    points.append("Sharpe ranking: " + f["ranking"] + ".")
    return points

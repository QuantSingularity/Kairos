import numpy as np
import pandas as pd

import config


def load_results(paths=config.PATHS):
    t = paths.tables
    res = {
        "quality": pd.read_csv(paths.processed / "data_quality.csv"),
        "regime_validation": pd.read_csv(t / "regime_validation.csv"),
        "models": pd.read_csv(t / "model_comparison.csv"),
        "summary": pd.read_csv(t / "summary_metrics.csv", index_col=0),
        "regime_metrics": pd.read_csv(t / "regime_metrics.csv", index_col=[0, 1]),
        "cost": pd.read_csv(t / "cost_sensitivity.csv", index_col=0),
        "subperiod": pd.read_csv(t / "subperiod_sharpe.csv", index_col=0),
        "significance": pd.read_csv(t / "significance.csv"),
    }
    return res


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
    direction = "higher" if row["diff"] > 0 else "lower"
    verdict = (
        "statistically significant at the 5% level"
        if row["p_value"] < 0.05
        else "not statistically significant"
    )
    return (
        f"{direction} Sharpe by {abs(row['diff']):.2f} (95% interval {row['ci_low']:.2f} to {row['ci_high']:.2f}, "
        f"p = {row['p_value']:.3f}), {verdict}"
    )


def findings(res):
    s, sig = res["summary"], res["significance"]
    out = {}
    ranked = s["Sharpe"].sort_values(ascending=False)
    out["ranking"] = ", ".join(f"{n} ({v:.2f})" for n, v in ranked.items())
    classical = [n for n in s.index if not n.startswith("BL_")]
    best_classical = s.loc[classical, "Sharpe"].idxmax()
    out["best_classical"] = best_classical
    for name in ("BL_ML_Regime", "BL_ML_Regime_IC"):
        sub = sig[sig["strategy"] == name]
        vs_best = sub[sub["benchmark"] == best_classical].iloc[0]
        out[f"{name}_vs_best"] = describe_diff(vs_best)
        classical_rows = sub[sub["benchmark"].isin(classical)]
        out[f"{name}_better"] = int(
            ((classical_rows["diff"] > 0) & (classical_rows["p_value"] < 0.05)).sum()
        )
        out[f"{name}_worse"] = int(
            ((classical_rows["diff"] < 0) & (classical_rows["p_value"] < 0.05)).sum()
        )
        out[f"{name}_n"] = len(classical_rows)
    ablation = sig[
        (sig["strategy"] == "BL_ML_Regime") & (sig["benchmark"] == "BL_ML_NoRegime")
    ].iloc[0]
    out["ablation"] = describe_diff(ablation)
    rm = res["regime_metrics"]
    stressed = rm.xs("Stressed", level=1)
    out["stress_cvar"] = (
        f"In stressed months the regime-aware portfolio's worst-5% average monthly return was "
        f"{stressed.loc['BL_ML_Regime', 'CVaR5']:.2%} against {stressed.loc['BL_ML_NoRegime', 'CVaR5']:.2%} without regime scaling."
    )
    m = res["models"].dropna(subset=["IC_ensemble_stacked"])
    out["mean_ic"] = float(m["IC_ensemble_stacked"].mean())
    out["sig_assets"] = int((m["p_value_stacked"] < 0.05).sum())
    out["n_assets"] = len(m)
    out["ens_beats"] = int(m["EnsembleBeatsMeanSingle"].sum())
    rv = res["regime_validation"]
    out["hmm_stress"] = float(rv.loc[rv["Kind"] == "stress", "Rate_hmm"].mean())
    out["hmm_calm"] = float(rv.loc[rv["Kind"] == "calm", "Rate_hmm"].mean())
    out["vol_stress"] = float(rv.loc[rv["Kind"] == "stress", "Rate_vol"].mean())
    out["vol_calm"] = float(rv.loc[rv["Kind"] == "calm", "Rate_vol"].mean())
    top = ranked.index[0]
    out["top"] = top
    out["answer"] = (
        f"The highest Sharpe ratio belongs to {top}. BL_ML_Regime is {out['BL_ML_Regime_vs_best']} relative to the best classical "
        f"benchmark ({best_classical}); it is significantly better than {out['BL_ML_Regime_better']} and significantly worse than "
        f"{out['BL_ML_Regime_worse']} of the {out['BL_ML_Regime_n']} classical benchmarks (5% level). BL_ML_Regime_IC is "
        f"{out['BL_ML_Regime_IC_vs_best']}."
    )
    return out

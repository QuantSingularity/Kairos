"""Headline numbers extracted from the result tables."""

from typing import Any

import numpy as np
import pandas as pd
from kairos import config
from kairos.reporting.results import describe_diff, require_pair

KAIROS_MAIN = ("BL_ML_Regime", "BL_ML_Regime_IC")


def skill_counts(m: pd.DataFrame) -> tuple[int, int]:
    """Negative R2 values and the number of values."""
    cols = config.MODELS + ["ensemble_equal", "ensemble_stacked"]
    r2 = m[[f"R2OOS_{c}" for c in cols if f"R2OOS_{c}" in m]].to_numpy().ravel()
    r2 = r2[~np.isnan(r2)]
    return int((r2 < 0).sum()), len(r2)


def findings(res: dict[str, Any]) -> dict[str, Any]:
    """Quantities quoted in the report and deck."""
    s, sig = res["summary"], res["significance"]
    out: dict[str, Any] = {}
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
        vs_best = require_pair(sig, name, best_classical)
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
    out["ablation"] = describe_diff(require_pair(sig, "BL_ML_Regime", "BL_ML_NoRegime"))
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
    out["r2_neg_pairs"], out["r2_pairs"] = skill_counts(m)
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
        f"the original stacked ensemble has a mean out-of-sample information coefficient of {out['mean_ic']:.3f}. "
        f"{out['sig_neg']} of {out['n_assets']} assets show an IC that is significantly negative at the 5% level"
        + (f" ({out['neg_assets']})" if out["sig_neg"] else "")
        + f" and {out['sig_pos']} show one that is significantly positive"
        + (f" ({out['pos_assets']})" if out["sig_pos"] else "")
        + f". The original stacked ensemble beats the mean of the single models on {out['ens_beats']} of {out['n_assets']} assets."
    )
    return out

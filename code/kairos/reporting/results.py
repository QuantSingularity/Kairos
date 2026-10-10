"""Loading of result tables and markdown helpers."""

from typing import Any

import numpy as np
import pandas as pd

from kairos import config


def load_results(paths: config.Paths = config.PATHS) -> dict[str, Any]:
    """Read every result table into one dictionary."""
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
        "skill_period": pd.read_csv(t / "skill_by_period.csv"),
        "stack_weights": pd.read_csv(t / "stack_weights.csv", index_col=0),
        "lstm_repair": pd.read_csv(t / "lstm_repair.csv"),
        "period": pd.read_csv(t / "period_split.csv", index_col=[0, 1]),
        "period_sig": pd.read_csv(t / "period_significance.csv"),
        "prior": pd.read_csv(t / "prior_comparison.csv"),
        "window": pd.read_csv(t / "window_robustness.csv", index_col=0),
        "window_sig": pd.read_csv(t / "window_significance.csv"),
        "window_meta": pd.read_csv(t / "window_meta.csv"),
        "holdout_status": pd.read_csv(t / "holdout_status.csv"),
        "holdout_metrics": (
            pd.read_csv(t / "holdout_metrics.csv", index_col=0)
            if (t / "holdout_metrics.csv").exists()
            else None
        ),
        "manifest": pd.read_csv(paths.processed / "data_manifest.csv"),
    }


def md_table(
    df: pd.DataFrame, formats: dict[str, str] | None = None, index: bool = True
) -> str:
    """Markdown table with per-column number formats."""
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


def describe_diff(row: pd.Series) -> str:
    """Sharpe difference, interval and verdict as text."""
    verdict = (
        "statistically significant at the 5% level"
        if row["p_value"] < 0.05
        else "not statistically significant"
    )
    return (
        f"{row['diff']:+.2f} (95% interval {row['ci_low']:.2f} to {row['ci_high']:.2f}, "
        f"p = {row['p_value']:.3f}), {verdict}"
    )


def require_pair(sig: pd.DataFrame, a: str, b: str) -> pd.Series:
    """Significance row for a pair, raising when the table lacks it."""
    row = pair(sig, a, b)
    if row is None:
        raise ValueError(f"missing significance row for {a} versus {b}")
    return row


def pair(sig: pd.DataFrame, a: str, b: str) -> pd.Series | None:
    """Significance row for a strategy and benchmark pair."""
    sub = sig[(sig["strategy"] == a) & (sig["benchmark"] == b)]
    return sub.iloc[0] if len(sub) else None

"""Parameter and prior sensitivity runs."""

import logging
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from kairos import config
from kairos.backtest import strategies as st
from kairos.backtest.engine import (
    monthly_data,
    naive_forecasts,
    performance,
    portfolio_returns,
)
from kairos.backtest.stats import paired_sharpe_test
from kairos.data.store import load_features, load_panel
from kairos.models.black_litterman import (
    black_litterman_weights,
    build_universe,
    default_params,
    load_prior_table,
)
from kairos.models.forecasting import load_forecasts

log = logging.getLogger(__name__)

VARIANTS = (("BL_ML_Regime", False), ("BL_ML_Regime_IC", True))


def run_grid(
    forecasts: dict[str, pd.DataFrame],
    panel: pd.DataFrame,
    regime: pd.Series,
    grid: dict[str, list[Any]] | None = None,
) -> pd.DataFrame:
    """Change one Black-Litterman parameter at a time."""
    settings = grid or config.SENSITIVITY_GRID
    _, log_rets, rf = monthly_data(panel)
    simple = np.expm1(log_rets)
    universe = build_universe(forecasts, log_rets)
    ref = performance(
        portfolio_returns(st.static_60_40(universe), simple, rf, config.COST_BPS)
    )
    base, rows = default_params(), []
    for name, values in settings.items():
        for value in values:
            if value == "observed":
                continue
            params = replace(base, **{name: value})
            for variant, ic in VARIANTS:
                w = black_litterman_weights(
                    forecasts, log_rets, rf, regime, universe, True, ic, params
                )
                m = performance(portfolio_returns(w, simple, rf, config.COST_BPS))
                rows.append(
                    {
                        "Parameter": name,
                        "Value": str(value),
                        "IsDefault": getattr(base, name) == value,
                        "Variant": variant,
                        "Sharpe": m["Sharpe"],
                        "AnnVol": m["AnnVol"],
                        "MaxDrawdown": m["MaxDrawdown"],
                        "AnnTurnover": m["AnnTurnover"],
                        "AvgCash": float(w["CASH"].mean()),
                        "Sharpe_60_40": ref["Sharpe"],
                    }
                )
    return pd.DataFrame(rows)


def run_priors(
    forecasts: dict[str, pd.DataFrame],
    panel: pd.DataFrame,
    regime: pd.Series,
    table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compare each prior with the portfolio built on it."""
    _, log_rets, rf = monthly_data(panel)
    simple = np.expm1(log_rets)
    universe = build_universe(forecasts, log_rets)
    kinds = ["reference", "equal", "inverse_vol"] + (
        ["observed"] if table is not None else []
    )
    naive = naive_forecasts(forecasts)
    variants = (
        ("BL_ML_Regime", forecasts, False),
        ("BL_ML_Regime_IC", forecasts, True),
        (config.NAIVE, naive, False),
    )
    rows = []
    for kind in kinds:
        params = replace(default_params(), prior=kind)
        matched = portfolio_returns(
            st.prior_portfolio(universe, kind, log_rets, table),
            simple,
            rf,
            config.COST_BPS,
        )
        base = performance(matched)
        for name, source, ic in variants:
            w = black_litterman_weights(
                source, log_rets, rf, regime, universe, True, ic, params, None, table
            )
            bt = portfolio_returns(w, simple, rf, config.COST_BPS)
            m = performance(bt)
            test = paired_sharpe_test(
                bt["net"] - bt["rf"], matched["net"] - matched["rf"]
            )
            rows.append(
                {
                    "Prior": kind,
                    "Variant": name,
                    "Sharpe": m["Sharpe"],
                    "Sharpe_PriorPortfolio": base["Sharpe"],
                    "Diff": test["diff"],
                    "CI_low": test["ci_low"],
                    "CI_high": test["ci_high"],
                    "p_value": test["p_value"],
                    "MaxDrawdown": m["MaxDrawdown"],
                    "MaxDrawdown_PriorPortfolio": base["MaxDrawdown"],
                }
            )
    return pd.DataFrame(rows)


def run(paths: config.Paths = config.PATHS) -> pd.DataFrame:
    """Sensitivity stage."""
    panel = load_panel(paths)
    F = load_features(paths)
    forecasts = load_forecasts(paths)
    table = run_grid(forecasts, panel, F["regime"])
    table.to_csv(paths.tables / "sensitivity.csv", index=False)
    priors = run_priors(forecasts, panel, F["regime"], load_prior_table(paths))
    priors.to_csv(paths.tables / "prior_comparison.csv", index=False)
    log.info("sensitivity grid: %d rows, priors: %d rows", len(table), len(priors))
    return table

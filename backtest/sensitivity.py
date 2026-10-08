from dataclasses import replace

import numpy as np
import pandas as pd

import config
from backtest import strategies as st
from backtest.engine import monthly_data, performance, portfolio_returns
from models.black_litterman import (
    black_litterman_weights,
    build_universe,
    default_params,
)
from models.forecasting import load_forecasts

VARIANTS = (("BL_ML_Regime", False), ("BL_ML_Regime_IC", True))


def run_grid(forecasts, panel, regime, grid=None):
    grid = grid or config.SENSITIVITY_GRID
    _, log_rets, rf = monthly_data(panel)
    simple = np.expm1(log_rets)
    universe = build_universe(forecasts, log_rets)
    ref = performance(
        portfolio_returns(st.static_60_40(universe), simple, rf, config.COST_BPS)
    )
    base, rows = default_params(), []
    for name, values in grid.items():
        for value in values:
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


def run(paths=config.PATHS):
    panel = pd.read_csv(
        paths.processed / "panel_daily.csv", index_col=0, parse_dates=True
    )
    F = pd.read_csv(
        paths.processed / "features_monthly.csv", index_col=0, parse_dates=True
    )
    table = run_grid(load_forecasts(paths), panel, F["regime"])
    table.to_csv(paths.tables / "sensitivity.csv", index=False)
    return table

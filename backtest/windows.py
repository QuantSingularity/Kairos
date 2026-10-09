import shutil
from contextlib import contextmanager

import pandas as pd

import config
from backtest import engine
from models import forecasting

KEEP = ("panel_daily.csv", "regimes_daily.csv", "features_monthly.csv")
KEY_PAIRS = (
    ("BL_ML_Regime", "Static_60_40"),
    ("BL_ML_Regime", "Regime_60_40"),
    ("Regime_60_40", "Static_60_40"),
)


@contextmanager
def initial_train(months):
    old = config.INITIAL_TRAIN_MONTHS
    config.INITIAL_TRAIN_MONTHS = months
    try:
        yield
    finally:
        config.INITIAL_TRAIN_MONTHS = old


def run(paths=config.PATHS):
    alt = config.Paths(paths.root / "results" / "alt_window")
    alt.ensure()
    for name in KEEP:
        shutil.copy(paths.processed / name, alt.processed / name)
    with initial_train(config.CRISIS_INITIAL_TRAIN):
        forecasting.run(alt, None, 1)
        engine.run(alt)
    main = pd.read_csv(paths.tables / "summary_metrics.csv", index_col=0)
    other = pd.read_csv(alt.tables / "summary_metrics.csv", index_col=0)
    ret = pd.read_csv(
        alt.processed / "backtest_returns.csv", index_col=0, parse_dates=True
    )
    table = pd.DataFrame(
        {
            "Sharpe_main": main["Sharpe"],
            "Sharpe_crisis_window": other["Sharpe"],
            "MaxDD_main": main["MaxDrawdown"],
            "MaxDD_crisis_window": other["MaxDrawdown"],
            "Months_main": main["Months"],
            "Months_crisis_window": other["Months"],
        }
    )
    table.attrs["start"] = str(ret.index[0].date())
    table.to_csv(paths.tables / "window_robustness.csv")
    sig = pd.read_csv(alt.tables / "significance.csv")
    keep = sig[[(a, b) in KEY_PAIRS for a, b in zip(sig["strategy"], sig["benchmark"])]]
    keep.to_csv(paths.tables / "window_significance.csv", index=False)
    pd.DataFrame(
        {
            "first_decision": [ret.index[0].date()],
            "last_decision": [ret.index[-1].date()],
            "initial_train_months": [config.CRISIS_INITIAL_TRAIN],
        }
    ).to_csv(paths.tables / "window_meta.csv", index=False)
    return table

"""Crisis-inclusive evaluation window."""

import logging
import shutil

import pandas as pd

from kairos import config
from kairos.backtest import engine
from kairos.data.store import load_returns
from kairos.models import forecasting

log = logging.getLogger(__name__)

KEEP = ("panel_daily.csv", "regimes_daily.csv", "features_monthly.csv")
KEY_PAIRS = (
    ("BL_ML_Regime", "Static_60_40"),
    ("BL_ML_Regime", "Regime_60_40"),
    ("Regime_60_40", "Static_60_40"),
)


def run(paths: config.Paths = config.PATHS) -> pd.DataFrame:
    """Re-run forecasts and backtest with a longer pre-2008 window."""
    alt = config.Paths(paths.results / "alt_window")
    alt.ensure()
    for name in KEEP:
        shutil.copy(paths.processed / name, alt.processed / name)
    log.info("crisis window with %d initial months", config.CRISIS_INITIAL_TRAIN)
    forecasting.run(alt, None, 1, initial_train=config.CRISIS_INITIAL_TRAIN)
    engine.run(alt)
    main = pd.read_csv(paths.tables / "summary_metrics.csv", index_col=0)
    other = pd.read_csv(alt.tables / "summary_metrics.csv", index_col=0)
    ret = load_returns(alt)
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

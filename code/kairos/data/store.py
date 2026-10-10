import pandas as pd

from kairos import config


def read_dated(path):
    return pd.read_csv(path, index_col=0, parse_dates=True)


def load_panel(paths=config.PATHS):
    return read_dated(paths.processed / "panel_daily.csv")


def load_features(paths=config.PATHS):
    return read_dated(paths.processed / "features_monthly.csv")


def load_regimes(paths=config.PATHS):
    return read_dated(paths.processed / "regimes_daily.csv")


def load_returns(paths=config.PATHS):
    return read_dated(paths.processed / "backtest_returns.csv")

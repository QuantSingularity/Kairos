"""Readers for the processed CSV files."""

from pathlib import Path

import pandas as pd

from kairos import config


def read_dated(path: Path) -> pd.DataFrame:
    """Read a CSV whose first column is a date index."""
    return pd.read_csv(path, index_col=0, parse_dates=True)


def load_panel(paths: config.Paths = config.PATHS) -> pd.DataFrame:
    """Daily price panel."""
    return read_dated(paths.processed / "panel_daily.csv")


def load_features(paths: config.Paths = config.PATHS) -> pd.DataFrame:
    """Monthly feature matrix."""
    return read_dated(paths.processed / "features_monthly.csv")


def load_regimes(paths: config.Paths = config.PATHS) -> pd.DataFrame:
    """Daily regime labels and probabilities."""
    return read_dated(paths.processed / "regimes_daily.csv")


def load_returns(paths: config.Paths = config.PATHS) -> pd.DataFrame:
    """Net monthly strategy returns."""
    return read_dated(paths.processed / "backtest_returns.csv")

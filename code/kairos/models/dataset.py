"""Feature selection and target construction for forecasting."""

import numpy as np
import pandas as pd

LSTM_FEATURES = [
    "ret_21d",
    "vol_21d",
    "rsi_14",
    "macd_hist_norm",
    "price_to_sma200",
    "mom_12_1",
    "drawdown_252d",
]


SHARED_FEATURES = ["VIX_level", "VIX_chg_5d", "VIX_zscore_252d", "regime", "p_stressed"]


def feature_columns(F: pd.DataFrame, asset: str) -> list[str]:
    """Asset-specific and shared feature names."""
    return [c for c in F.columns if c.startswith(asset + "_")] + [
        c for c in SHARED_FEATURES if c in F.columns
    ]


def asset_frame(F: pd.DataFrame, asset: str) -> pd.DataFrame:
    """Rows with a usable feature history for the asset."""
    rows = F[F[f"{asset}_mom_12_1"].notna() & F["regime"].notna()]
    return rows


def design_matrix(rows: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Finite feature matrix with missing values set to zero."""
    return rows[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values


def forward_targets(log_price: pd.Series) -> pd.Series:
    """Next-month log return."""
    return log_price.shift(-1) - log_price


def historical_mean(target: pd.Series) -> pd.Series:
    """Expanding mean known at the forecast date."""
    return target.expanding().mean().shift(1)

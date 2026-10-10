"""Causal feature engineering."""

import logging

import numpy as np
import pandas as pd

from kairos import config
from kairos.data.loader import month_end_dates
from kairos.data.store import load_panel, load_regimes

log = logging.getLogger(__name__)


def rsi(price: pd.Series, window: int = 14) -> pd.Series:
    """Relative strength index."""
    delta = price.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    out = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    out = out.where(loss != 0, 100.0)
    return out.where(gain.notna() & loss.notna())


def asset_features(price: pd.Series, spy_ret: pd.Series) -> pd.DataFrame:
    """Daily technical features for one asset."""
    log_price = np.log(price)
    ret1 = log_price.diff()
    f = {}
    for h in (1, 5, 10, 21, 63):
        f[f"ret_{h}d"] = log_price.diff(h)
    for w in (10, 21, 63):
        f[f"vol_{w}d"] = ret1.rolling(w).std() * np.sqrt(252)
    sma = {w: price.rolling(w).mean() for w in (20, 50, 200)}
    f["sma_ratio_20_50"] = sma[20] / sma[50] - 1
    f["sma_ratio_50_200"] = sma[50] / sma[200] - 1
    f["price_to_sma200"] = price / sma[200] - 1
    f["mom_12_1"] = log_price.shift(21) - log_price.shift(252)
    f["drawdown_252d"] = price / price.rolling(252, min_periods=60).max() - 1
    f["rsi_14"] = rsi(price)
    macd = (
        price.ewm(span=12, adjust=False).mean()
        - price.ewm(span=26, adjust=False).mean()
    )
    signal = macd.ewm(span=9, adjust=False).mean()
    f["macd_norm"] = macd / price
    f["macd_hist_norm"] = (macd - signal) / price
    f["beta_63d_spy"] = ret1.rolling(63).cov(spy_ret) / spy_ret.rolling(63).var()
    f["corr_63d_spy"] = ret1.rolling(63).corr(spy_ret)
    out = pd.DataFrame(f)
    out.columns = [f"{price.name}_{c}" for c in out.columns]
    return out


def build_daily_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Daily features for every asset plus VIX inputs."""
    spy_ret = np.log(panel["SPY"]).diff()
    frames = []
    for asset in config.ASSETS:
        if asset in panel:
            price = panel[asset].dropna().rename(asset)
            frames.append(asset_features(price, spy_ret))
    vix = panel["VIX"]
    frames.append(
        pd.DataFrame(
            {
                "VIX_level": vix,
                "VIX_chg_5d": vix.pct_change(5),
                "VIX_zscore_252d": (vix - vix.rolling(252).mean())
                / vix.rolling(252).std(),
            }
        )
    )
    return pd.concat(frames, axis=1).replace([np.inf, -np.inf], np.nan)


def monthly_features(panel: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    """Month-end feature matrix with regime columns."""
    dates = month_end_dates(panel.index)
    daily = build_daily_features(panel)
    F = daily.reindex(dates)
    F["regime"] = regimes["regime"].reindex(dates)
    F["p_stressed"] = regimes["p_stressed"].reindex(dates)
    return F


def run(paths: config.Paths = config.PATHS) -> pd.DataFrame:
    """Features stage."""
    panel = load_panel(paths)
    regimes = load_regimes(paths)
    F = monthly_features(panel, regimes)
    F.to_csv(paths.processed / "features_monthly.csv")
    log.info("features: %d months, %d columns", *F.shape)
    return F

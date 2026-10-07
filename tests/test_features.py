import numpy as np
import pandas as pd
from fixtures import make_panel

from features.build_features import asset_features, build_daily_features, rsi


def test_features_are_causal():
    panel = make_panel(start="2005-01-01", end="2010-12-31").dropna(
        subset=["SPY", "EFA", "TLT", "IEF"]
    )
    full = build_daily_features(panel)
    cut = 900
    part = build_daily_features(panel.iloc[:cut])
    pd.testing.assert_frame_equal(full.iloc[:cut], part, check_exact=False, atol=1e-10)


def test_no_price_level_features():
    panel = make_panel(start="2005-01-01", end="2008-12-31").dropna(subset=["SPY"])
    f = asset_features(panel["SPY"].rename("SPY"), np.log(panel["SPY"]).diff())
    assert not any(
        c.endswith(("_sma_20", "_sma_50", "_sma_200", "_macd", "_macd_signal"))
        for c in f.columns
    )
    assert (
        f.drop(columns=["SPY_vol_10d", "SPY_vol_21d", "SPY_vol_63d", "SPY_rsi_14"])
        .abs()
        .max()
        .max()
        < 50
    )


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.arange(1.0, 60.0))
    assert rsi(up).dropna().eq(100.0).all()
    down = pd.Series(np.arange(60.0, 1.0, -1.0))
    assert rsi(down).dropna().eq(0.0).all()
    mixed = pd.Series(np.random.default_rng(0).normal(0, 1, 200).cumsum() + 100)
    r = rsi(mixed).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_momentum_skips_last_month():
    price = pd.Series(np.exp(np.linspace(0, 1, 400)), name="X")
    f = asset_features(price, np.log(price).diff())
    t = 350
    expected = np.log(price.iloc[t - 21]) - np.log(price.iloc[t - 252])
    assert abs(f["X_mom_12_1"].iloc[t] - expected) < 1e-12

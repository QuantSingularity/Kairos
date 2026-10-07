import numpy as np
import pandas as pd
import pytest
from fixtures import make_panel, write_raw_cache

import config
from data import loader


@pytest.fixture(scope="module")
def panel():
    return make_panel(start="2003-01-01", end="2016-12-30")


def test_build_panel_from_cache(tmp_path, panel):
    paths = config.Paths(tmp_path)
    write_raw_cache(paths, panel)
    raw = loader.fetch_prices(paths)
    built = loader.build_panel(raw, end="2016-12-30")
    assert set(config.REQUIRED) <= set(built.columns)
    assert built["RF"].max() < 0.2
    assert built.index.is_monotonic_increasing


def test_validate_accepts_good_panel(panel):
    report, problems = loader.validate_panel(panel.dropna(subset=config.REQUIRED))
    assert problems == []
    assert "corr(SPY,EFA)" in set(report["series"])


def test_validate_rejects_uncorrelated_proxy(panel):
    bad = panel.dropna(subset=config.REQUIRED).copy()
    rng = np.random.default_rng(1)
    bad["EFA"] = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(bad))))
    _, problems = loader.validate_panel(bad)
    assert any("SPY and EFA" in p for p in problems)


def test_validate_rejects_extreme_move(panel):
    bad = panel.dropna(subset=config.REQUIRED).copy()
    bad.iloc[500, bad.columns.get_loc("SPY")] *= 3
    _, problems = loader.validate_panel(bad)
    assert any("SPY" in p and "daily move" in p for p in problems)


def test_validate_rejects_short_history(panel):
    short = panel.dropna(subset=config.REQUIRED).iloc[:600]
    _, problems = loader.validate_panel(short)
    assert any("fewer than" in p for p in problems)


def test_month_end_dates_drops_partial_month():
    idx = pd.bdate_range("2020-01-01", "2020-03-18")
    dates = loader.month_end_dates(idx)
    assert dates[-1] == pd.Timestamp("2020-02-28")
    assert len(loader.month_end_dates(idx, drop_partial=False)) == 3


def test_download_failure_is_reported(monkeypatch):
    import sys
    import types

    fake = types.SimpleNamespace(
        Ticker=lambda t: types.SimpleNamespace(history=lambda **k: pd.DataFrame())
    )
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    monkeypatch.setattr(loader.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError):
        loader.download_ticker("SPY", "2020-01-01", "2020-02-01", retries=2)

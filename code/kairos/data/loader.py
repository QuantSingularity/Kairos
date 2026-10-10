"""Price download, caching, panel construction and validation."""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from kairos import config

log = logging.getLogger(__name__)

CORE_CHECKS = [("SPY", "EFA", 0.6), ("TLT", "IEF", 0.7), ("SPY", "HYG", 0.4)]


def cache_file(paths: config.Paths, ticker: str) -> Path:
    """Path of the cached CSV for a ticker."""
    return paths.raw / (ticker.replace("^", "").replace("-", "_") + ".csv")


def download_ticker(ticker: str, start: str, end: str, retries: int = 3) -> pd.Series:
    """Download adjusted closes for one ticker with retries."""
    import yfinance as yf

    last = None
    for attempt in range(retries):
        try:
            hist = yf.Ticker(ticker).history(
                start=start, end=end, auto_adjust=True, actions=False
            )
            if len(hist):
                s = hist["Close"].copy()
                s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
                s = s[~s.index.duplicated(keep="last")].sort_index()
                s.name = ticker
                return s
        except Exception as exc:
            last = exc
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"could not download {ticker} from Yahoo Finance: {last}")


def fetch_prices(
    paths: config.Paths = config.PATHS,
    refresh: bool = False,
    start: str = config.START_DATE,
    end: str = config.END_DATE,
) -> pd.DataFrame:
    """Load cached prices, downloading missing series."""
    paths.raw.mkdir(parents=True, exist_ok=True)
    tickers = config.ASSETS + [config.VIX_TICKER, config.RF_TICKER]
    series = {}
    for ticker in tickers:
        path = cache_file(paths, ticker)
        if path.exists() and not refresh:
            s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
            s.name = ticker
        else:
            s = download_ticker(ticker, start, end)
            s.to_csv(path, header=True)
        series[ticker] = s
    return pd.DataFrame(series)


def build_panel(raw: pd.DataFrame, end: str = config.END_DATE) -> pd.DataFrame:
    """Align the raw series into the daily panel."""
    raw = raw.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    raw = raw.rename(columns={config.VIX_TICKER: "VIX", config.RF_TICKER: "RF"})
    last = min(raw.index.max(), pd.Timestamp(end))
    index = pd.bdate_range(raw.index.min(), last)
    panel = raw.reindex(index).ffill(limit=5)
    panel["RF"] = panel["RF"] / 100.0
    panel = panel.dropna(subset=config.REQUIRED)
    return panel[[c for c in config.ASSETS if c in panel] + ["VIX", "RF"]]


def month_end_dates(
    index: pd.DatetimeIndex, drop_partial: bool = True
) -> pd.DatetimeIndex:
    """Last trading day of each month."""
    index = pd.DatetimeIndex(index)
    ends = pd.Series(index, index=index).groupby(index.to_period("M")).max()
    dates = pd.DatetimeIndex(ends.values)
    if drop_partial and len(dates):
        if dates[-1] < dates[-1] + pd.offsets.BMonthEnd(0):
            dates = dates[:-1]
    return dates


def validate_panel(
    panel: pd.DataFrame, min_years: float = 8.0
) -> tuple[pd.DataFrame, list[str]]:
    """Quality report and the list of blocking problems."""
    rows, problems = [], []
    daily = panel.pct_change()
    for col in panel.columns:
        s = panel[col].dropna()
        limit = 0.8 if col in config.CRYPTO else 0.6
        if not len(s) and col not in config.REQUIRED:
            rows.append({"series": col, "first_date": None, "observations": 0})
            continue
        max_move = float(daily[col].abs().max()) if col not in ("VIX", "RF") else np.nan
        gaps = float(panel[col].loc[s.index[0] :].isna().mean()) if len(s) else 1.0
        rows.append(
            {
                "series": col,
                "first_date": s.index[0].date() if len(s) else None,
                "observations": len(s),
                "missing_after_start": gaps,
                "max_abs_daily_move": max_move,
            }
        )
        if col in config.REQUIRED and len(s) < min_years * 252:
            problems.append(f"{col} has fewer than {min_years} years of history")
        if not np.isnan(max_move) and max_move > limit:
            problems.append(
                f"{col} has a daily move of {max_move:.0%}, likely a data error"
            )
        if gaps > 0.02:
            problems.append(f"{col} has {gaps:.1%} missing values after its first date")
    monthly = np.log(
        panel[[c for c in config.ASSETS if c in panel]].resample("ME").last()
    ).diff()
    for a, b, floor in CORE_CHECKS:
        if a in monthly and b in monthly:
            corr = monthly[a].corr(monthly[b])
            rows.append(
                {
                    "series": f"corr({a},{b})",
                    "observations": int(monthly[[a, b]].dropna().shape[0]),
                    "value": corr,
                }
            )
            if corr < floor:
                problems.append(
                    f"monthly correlation of {a} and {b} is {corr:.2f}, below {floor}"
                )
    report = pd.DataFrame(rows)
    return report, problems


def manifest(paths: config.Paths = config.PATHS) -> pd.DataFrame:
    """Row counts, date ranges and hashes of the cached series."""
    import hashlib

    rows = []
    for ticker in config.ASSETS + [config.VIX_TICKER, config.RF_TICKER]:
        path = cache_file(paths, ticker)
        if not path.exists():
            continue
        s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0].dropna()
        if s.empty:
            continue
        rows.append(
            {
                "series": ticker,
                "file": path.name,
                "rows": len(s),
                "first": s.index[0].date(),
                "last": s.index[-1].date(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
            }
        )
    return pd.DataFrame(rows)


def run(
    paths: config.Paths = config.PATHS, refresh: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Data stage."""
    paths.ensure()
    panel = build_panel(fetch_prices(paths, refresh))
    manifest(paths).to_csv(paths.processed / "data_manifest.csv", index=False)
    report, problems = validate_panel(panel)
    report.to_csv(paths.processed / "data_quality.csv", index=False)
    if problems:
        raise ValueError("panel failed validation: " + "; ".join(problems))
    panel.to_csv(paths.processed / "panel_daily.csv")
    log.info(
        "panel: %d days, %s to %s",
        len(panel),
        panel.index[0].date(),
        panel.index[-1].date(),
    )
    return panel, report

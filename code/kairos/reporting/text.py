"""Static report text."""

from kairos import config


def fmt_prior() -> str:
    """Reference prior weights as text."""
    w = sorted(config.MARKET_WEIGHTS.items(), key=lambda kv: -kv[1])
    return ", ".join(f"{k} {v:.0%}" for k, v in w)


def data_intro() -> list[str]:
    """Data source description."""
    return [
        "All prices are daily adjusted closes downloaded from Yahoo Finance through the yfinance library, with the CBOE VIX (^VIX) "
        "and the 13-week Treasury bill yield (^IRX) as the volatility and risk-free inputs. No series is simulated, interpolated "
        "or reconstructed. The panel is validated before use: history length, gaps, extreme daily moves, and sanity floors on "
        "cross-asset correlations (for example SPY and EFA) that would expose substitute or corrupted data. The manifest below "
        "records the row count, date range and a file hash for every cached series, so a fresh download can be checked against "
        "this run.",
    ]


def universe_note() -> list[str]:
    """Note on the expanding universe."""
    return [
        "The investable universe expands over time. An asset enters the portfolio only after it has a valid walk-forward forecast and "
        f"at least {config.MIN_HISTORY} months of return history, so crypto assets enter when they have a usable track record.",
    ]


def method_section() -> list[str]:
    """Method section."""
    return [
        "## 4. Method",
        "**Features.** About 20 causal technical, volatility, momentum, drawdown and market-beta features per asset, computed from daily "
        "prices, plus VIX level, change and z-score. Raw price levels are never used as features. Features are sampled at month end.",
        "**Regimes.** A two-state Gaussian HMM on SPY return, 21-day realised volatility and VIX. It is refit once a year on past data "
        "only, and daily states come from the forward filter, so the label on any day uses no later information. A volatility-percentile "
        "rule is the comparison method.",
        "**Forecasting.** One-month-ahead log return per asset. Candidates are walk-forward XGBoost, Random Forest and Ridge per asset, "
        "an LSTM written in NumPy whose training target is scaled by its training mean and standard deviation, two pooled cross-asset "
        "models (ridge and XGBoost, trained on all assets' deviations from their own expanding mean), and the expanding historical mean. "
        f"The original design stacked the first four models with non-negative weights fit on the previous {config.STACK_WINDOW} months "
        f"(ensemble_stacked). The final forecast stacks all candidates with convex weights (non-negative, summing to one) and then "
        f"shrinks the result toward the historical mean by the slope of realised on predicted deviations over the previous "
        f"{config.CALIBRATION_WINDOW} months, clipped to the range zero to one and set to zero until {config.CALIBRATION_MIN} months "
        f"exist. The first {config.INITIAL_TRAIN_MONTHS} months are used only for initial training. Skill is judged by the "
        "information coefficient, the hit rate against the always-up baseline, and the out-of-sample R2 against the expanding "
        "historical mean.",
        "**Forecast diagnostics.** The tree and linear models are re-fitted without the regime features, without all market-wide "
        f"stress inputs (regime and VIX), with a {config.DIAG_INITIAL_TRAIN}-month initial window, and with the first and last changes "
        "combined, and compared on the dates common to all designs. The link between the stress probability and next-month returns "
        "is measured separately inside and after the initial window. The LSTM is also re-run without target scaling to measure the "
        "effect of the repair.",
        "**Portfolio construction.** Black-Litterman with a Ledoit-Wolf covariance, a prior portfolio, and the calibrated forecasts as "
        "absolute views. View uncertainty follows the trailing forecast error variance. In the stressed regime, view uncertainty is "
        "multiplied and risk aversion is raised. The optimiser allows cash, so higher risk aversion lowers exposure. "
        "BL_ML_Regime_IC additionally scales each view by the asset's trailing IC. BL_ML_NoRegime removes the regime scaling and "
        "serves as the ablation. BL_ML_Raw_Regime uses the original uncalibrated stacked forecasts, and BL_HistMean_Regime uses the "
        "expanding historical mean as the view.",
        "**Priors.** The default prior is a hand-picked reference allocation. Alternatives are an equal-weight prior and a trailing "
        "inverse-volatility prior, which need no judgement, and an observed prior read from `data/raw/prior_weights.csv` when that file "
        "exists. Each Black-Litterman portfolio is compared with the portfolio that holds its own prior.",
        "**Benchmarks.** Market reference weights, equal weight, 60/40 (SPY and IEF), inverse volatility, and mean-variance with Ledoit-Wolf "
        "covariance. All benchmarks use the same eligible universe as the Black-Litterman portfolios. Regime_60_40 and Regime_Market "
        "apply the stress rule mechanically to 60/40 and to the market reference, with no forecasts.",
        "**Backtest.** Monthly rebalancing, 10 basis points on traded notional, weight drift, cash earning the T-bill rate, and Sharpe ratios "
        "measured on excess returns over that rate. Significance uses a paired stationary bootstrap of the Sharpe difference, centred on the "
        "null. A deflated Sharpe ratio allows for the Kairos variants tried. Sensitivity runs change one Black-Litterman parameter at a time.",
        "**Evaluation windows.** Results are reported for the full sample, for the periods before and from "
        f"{config.HOLDOUT_START}, and for a crisis-inclusive window with a {config.CRISIS_INITIAL_TRAIN}-month initial training period that "
        "starts the out-of-sample forecasts before 2008. A specification hash and freeze date are stored so that months realised after "
        "the freeze can be evaluated without any re-tuning.",
    ]


def closing_section() -> list[str]:
    """Limitations and reproducibility."""
    return [
        "## 7. Limitations",
        "Monthly data gives roughly 15 years of observations in the main window and about 20 in the crisis-inclusive window, which "
        "limits statistical power, so wide intervals should be read as inconclusive rather than as evidence of equivalence. "
        f"The default prior is a hand-picked reference allocation ({fmt_prior()}), not an observed market-capitalisation series. "
        "The inverse-volatility and equal-weight priors remove the judgement but are not market-implied either. "
        "The stress rule is calibrated by hand (multipliers fixed in advance) and the validation windows were chosen with hindsight, "
        "although they are not used in fitting. The forecast repairs and the diagnostics were designed after seeing full-sample "
        "results, so the development and holdout split is a reporting device and only months realised after the freeze date are "
        "untouched evidence. The LSTM is a small NumPy implementation refit every six months. Yahoo adjusted closes are "
        "retrospective vendor data and can be revised. The data stage needs network access to Yahoo Finance for a fresh download; "
        "later runs reuse the cache, and the manifest in section 3 allows the cached series to be checked. Results hold for this "
        "universe and period.",
        "## 8. Reproducibility",
        "Run `python scripts/run.py all` to download data, run every stage and rebuild this report, or `python scripts/run.py all --refresh` to "
        "download fresh prices first. Seeds, parameters and paths are in `code/kairos/config.py`.",
    ]

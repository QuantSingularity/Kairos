"""Walk-forward hidden Markov regime detection."""

import logging
from typing import Any

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from kairos import config
from kairos.data.store import load_panel
from scipy.special import logsumexp

log = logging.getLogger(__name__)

STRESS_PERIODS = {
    "2008 Global Financial Crisis": ("2008-09-01", "2009-03-31"),
    "2011 European Debt Crisis": ("2011-07-15", "2011-10-15"),
    "2015-2016 China Growth Scare": ("2015-08-15", "2016-02-15"),
    "2018 Q4 Selloff": ("2018-10-01", "2018-12-24"),
    "2020 COVID Crash": ("2020-02-19", "2020-04-07"),
    "2022 Rate-Hiking Selloff": ("2022-01-03", "2022-10-14"),
}
CALM_PERIODS = {
    "2013-2014 Calm": ("2013-01-02", "2014-12-31"),
    "2017 Low Volatility": ("2017-01-03", "2017-12-29"),
    "2021 Recovery": ("2021-03-01", "2021-11-30"),
}


def regime_inputs(panel: pd.DataFrame) -> pd.DataFrame:
    """Return, realised volatility and VIX inputs."""
    ret = np.log(panel["SPY"]).diff()
    return pd.DataFrame(
        {"ret": ret, "vol21": ret.rolling(21).std() * np.sqrt(252), "vix": panel["VIX"]}
    ).dropna()


def fit_hmm(
    X: np.ndarray, n_states: int = 2, seed: int = config.SEED, n_init: int = 3
) -> Any:
    """Best of several HMM fits by log likelihood."""
    best = None
    for k in range(n_init):
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=200,
            tol=1e-4,
            random_state=seed + k,
        )
        try:
            model.fit(X)
            score = model.score(X)
        except ValueError:
            continue
        if best is None or score > best[0]:
            best = (score, model)
    if best is None:
        raise RuntimeError("HMM fit failed for every initialisation")
    return best[1]


def forward_filter(model: Any, X: np.ndarray) -> np.ndarray:
    """Filtered state probabilities using past data only."""
    log_b = model._compute_log_likelihood(X)
    log_a = np.log(np.clip(model.transmat_, 1e-300, None))
    alpha = np.empty_like(log_b)
    alpha[0] = np.log(np.clip(model.startprob_, 1e-300, None)) + log_b[0]
    alpha[0] -= logsumexp(alpha[0])
    for t in range(1, len(X)):
        alpha[t] = logsumexp(alpha[t - 1][:, None] + log_a, axis=0) + log_b[t]
        alpha[t] -= logsumexp(alpha[t])
    return np.exp(alpha)


def walk_forward_hmm(
    inputs: pd.DataFrame,
    min_train: int = config.HMM_MIN_TRAIN,
    refit_every: int = config.HMM_REFIT_EVERY,
    seed: int = config.SEED,
) -> pd.Series:
    """Filtered stress probability with yearly refits."""
    X = inputs.values
    p = np.full(len(X), np.nan)
    for start in range(min_train, len(X), refit_every):
        stop = min(start + refit_every, len(X))
        mu, sd = X[:start].mean(axis=0), X[:start].std(axis=0)
        Z = (X[:stop] - mu) / sd
        model = fit_hmm(Z[:start], seed=seed)
        stressed = int(np.argmax(model.means_[:, 1]))
        p[start:stop] = forward_filter(model, Z)[start:stop, stressed]
    return pd.Series(p, index=inputs.index, name="p_stressed")


def volatility_regime(
    inputs: pd.DataFrame, window: int = 252, quantile: float = 0.8
) -> pd.Series:
    """Volatility percentile comparison rule."""
    vol = inputs["vol21"]
    threshold = vol.rolling(window, min_periods=60).quantile(quantile)
    return (vol > threshold).astype(float).where(threshold.notna())


def detect(
    panel: pd.DataFrame,
    min_train: int = config.HMM_MIN_TRAIN,
    refit_every: int = config.HMM_REFIT_EVERY,
) -> pd.DataFrame:
    """Daily regime labels and probabilities."""
    inputs = regime_inputs(panel)
    p = walk_forward_hmm(inputs, min_train, refit_every)
    out = pd.DataFrame({"p_stressed": p})
    out["regime"] = (p > config.STRESS_THRESHOLD).astype(float).where(p.notna())
    out["vol_regime"] = volatility_regime(inputs)
    return out


def validate(regime: pd.Series, periods: Any = None) -> pd.DataFrame:
    """Share of days flagged in known stress and calm periods."""
    periods = periods or {
        **{k: (v, 1) for k, v in STRESS_PERIODS.items()},
        **{k: (v, 0) for k, v in CALM_PERIODS.items()},
    }
    rows = []
    for name, ((start, end), kind) in periods.items():
        window = regime.loc[start:end].dropna()
        rows.append(
            {
                "Period": name,
                "Kind": "stress" if kind == 1 else "calm",
                "Days": len(window),
                "FlaggedStressed": int((window == 1).sum()),
                "Rate": float((window == 1).mean()) if len(window) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run(paths: config.Paths = config.PATHS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Regime stage."""
    panel = load_panel(paths)
    out = detect(panel)
    out.to_csv(paths.processed / "regimes_daily.csv")
    val_hmm = validate(out["regime"])
    val_vol = validate(out["vol_regime"])
    merged = val_hmm.merge(val_vol, on=["Period", "Kind"], suffixes=("_hmm", "_vol"))
    merged.to_csv(paths.tables / "regime_validation.csv", index=False)
    log.info("regimes written for %d days", len(out))
    return out, merged

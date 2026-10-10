"""Walk-forward fitting of the single-asset models."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from kairos import config
from kairos.models.lstm_numpy import LSTMRegressor


def make_model(name: str, seed: int) -> Any:
    """Build one of the tree or linear models."""
    if name == "xgb":
        return XGBRegressor(
            n_estimators=60,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=1,
            verbosity=0,
        )
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=4,
            min_samples_leaf=3,
            random_state=seed,
            n_jobs=1,
        )
    if name == "ridge":
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 4, 9)))
    raise ValueError(name)


def walk_forward_sklearn(
    X: np.ndarray,
    y: np.ndarray,
    names: list[str],
    seed: int = config.SEED,
    start: int | None = None,
) -> dict[str, np.ndarray]:
    """Expanding-window forecasts for the tree and linear models."""
    start = config.INITIAL_TRAIN_MONTHS if start is None else start
    n = len(y)
    preds = {m: np.full(n, np.nan) for m in names}
    for t in range(start, n):
        train = np.arange(t)
        train = train[~np.isnan(y[train])]
        for m in names:
            model = make_model(m, seed).fit(X[train], y[train])
            preds[m][t] = model.predict(X[t : t + 1])[0]
    return preds


def sequences(
    Z: np.ndarray, rows: Sequence[int] | np.ndarray, length: int
) -> np.ndarray:
    """Rolling input windows ending at each row."""
    return np.stack([Z[r - length + 1 : r + 1] for r in rows])


def walk_forward_lstm(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = config.SEED,
    start: int | None = None,
    hidden: int | None = None,
    lr: float | None = None,
    l2: float | None = None,
    scale_target: bool = True,
) -> np.ndarray:
    """Periodically refitted LSTM forecasts."""
    start = config.INITIAL_TRAIN_MONTHS if start is None else start
    n, L = len(y), config.LSTM_SEQ_LEN
    preds = np.full(n, np.nan)
    model, mu, sd, ym, ys = None, None, None, 0.0, 1.0
    for t in range(start, n):
        if model is None or (t - start) % config.LSTM_REFIT_EVERY == 0:
            mu, sd = X[:t].mean(axis=0), X[:t].std(axis=0) + 1e-8
            Z = (X - mu) / sd
            rows = np.array([r for r in range(L - 1, t) if not np.isnan(y[r])])
            if len(rows) < 30:
                continue
            cut = int(len(rows) * 0.85)
            ym, ys = (
                (y[rows[:cut]].mean(), y[rows[:cut]].std() + 1e-8)
                if scale_target
                else (0.0, 1.0)
            )
            model = LSTMRegressor(
                n_features=X.shape[1],
                hidden_size=hidden or config.LSTM_HIDDEN,
                seed=seed,
                lr=lr or config.LSTM_LR,
                l2=config.LSTM_L2 if l2 is None else l2,
                epochs=config.LSTM_EPOCHS,
                patience=config.LSTM_PATIENCE,
            )
            model.fit(
                sequences(Z, rows[:cut], L),
                (y[rows[:cut]] - ym) / ys,
                sequences(Z, rows[cut:], L),
                (y[rows[cut:]] - ym) / ys,
            )
        if model is not None:
            Z = (X - mu) / sd
            preds[t] = ym + ys * model.predict(sequences(Z, [t], L))[0]
    return preds

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import nnls
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import config
from data.loader import month_end_dates
from models.lstm_numpy import LSTMRegressor

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


def make_model(name, seed):
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


def feature_columns(F, asset):
    return [c for c in F.columns if c.startswith(asset + "_")] + [
        c for c in SHARED_FEATURES if c in F.columns
    ]


def asset_frame(F, asset):
    rows = F[F[f"{asset}_mom_12_1"].notna() & F["regime"].notna()]
    return rows


def forward_targets(log_price):
    return log_price.shift(-1) - log_price


def walk_forward_sklearn(X, y, names, seed=config.SEED, start=None):
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


def sequences(Z, rows, length):
    return np.stack([Z[r - length + 1 : r + 1] for r in rows])


def walk_forward_lstm(X, y, seed=config.SEED, start=None):
    start = config.INITIAL_TRAIN_MONTHS if start is None else start
    n, L = len(y), config.LSTM_SEQ_LEN
    preds = np.full(n, np.nan)
    model, mu, sd = None, None, None
    for t in range(start, n):
        if model is None or (t - start) % config.LSTM_REFIT_EVERY == 0:
            mu, sd = X[:t].mean(axis=0), X[:t].std(axis=0) + 1e-8
            Z = (X - mu) / sd
            rows = np.array([r for r in range(L - 1, t) if not np.isnan(y[r])])
            if len(rows) < 30:
                continue
            cut = int(len(rows) * 0.85)
            model = LSTMRegressor(
                n_features=X.shape[1],
                seed=seed,
                epochs=config.LSTM_EPOCHS,
                patience=config.LSTM_PATIENCE,
            )
            model.fit(
                sequences(Z, rows[:cut], L),
                y[rows[:cut]],
                sequences(Z, rows[cut:], L),
                y[rows[cut:]],
            )
        if model is not None:
            Z = (X - mu) / sd
            preds[t] = model.predict(sequences(Z, [t], L))[0]
    return preds


def stack_forecasts(P, y, window=None):
    window = config.STACK_WINDOW if window is None else window
    stacked = P.mean(axis=1, skipna=False).copy()
    values = P.values
    target = y.values
    for i in range(len(P)):
        if np.isnan(values[i]).any():
            continue
        hist = np.arange(i)
        ok = hist[~np.isnan(values[hist]).any(axis=1) & ~np.isnan(target[hist])]
        if len(ok) < window:
            continue
        ok = ok[-window:]
        weights, _ = nnls(values[ok], target[ok])
        stacked.iloc[i] = values[i] @ weights
    return P.mean(axis=1, skipna=False), stacked


def run_asset(asset, F, log_price, seed=config.SEED):
    rows = asset_frame(F, asset)
    y = forward_targets(log_price.reindex(rows.index)).values
    cols = feature_columns(F, asset)
    X = rows[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    preds = walk_forward_sklearn(X, y, ["xgb", "rf", "ridge"], seed)
    lstm_cols = [f"{asset}_{c}" for c in LSTM_FEATURES] + ["VIX_zscore_252d", "regime"]
    X_lstm = rows[lstm_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    preds["lstm"] = walk_forward_lstm(X_lstm, y, seed)
    out = pd.DataFrame({"target": y, **preds}, index=rows.index)
    equal, stacked = stack_forecasts(out[config.MODELS], out["target"])
    out["ensemble_equal"], out["ensemble_stacked"] = equal, stacked
    return out


def run_all(paths=config.PATHS, assets=None, n_jobs=-1, seed=config.SEED):
    F = pd.read_csv(
        paths.processed / "features_monthly.csv", index_col=0, parse_dates=True
    )
    panel = pd.read_csv(
        paths.processed / "panel_daily.csv", index_col=0, parse_dates=True
    )
    log_price = np.log(panel.reindex(month_end_dates(panel.index)))
    chosen = [
        a
        for a in (assets or config.ASSETS)
        if a in panel and len(asset_frame(F, a)) >= config.INITIAL_TRAIN_MONTHS + 24
    ]
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_asset)(a, F, log_price[a], seed) for a in chosen
    )
    for asset, res in zip(chosen, results):
        res.to_csv(paths.processed / f"forecasts_{asset.replace('-', '_')}.csv")
    return dict(zip(chosen, results))


def load_forecasts(paths=config.PATHS):
    out = {}
    for asset in config.ASSETS:
        path = paths.processed / f"forecasts_{asset.replace('-', '_')}.csv"
        if path.exists():
            out[asset] = pd.read_csv(path, index_col=0, parse_dates=True)
    return out


def evaluate(forecasts):
    rows = []
    cols = config.MODELS + ["ensemble_equal", "ensemble_stacked"]
    for asset, df in forecasts.items():
        oos = df.dropna(subset=["target"] + cols)
        row = {"Asset": asset, "n_oos": len(oos)}
        if len(oos) < 12:
            rows.append(row)
            continue
        for c in cols:
            ic, p = spearmanr(oos[c], oos["target"])
            row[f"IC_{c}"] = ic
            row[f"Hit_{c}"] = float((np.sign(oos[c]) == np.sign(oos["target"])).mean())
            if c == "ensemble_stacked":
                row["p_value_stacked"] = p
        singles = [row[f"IC_{m}"] for m in config.MODELS]
        row["MeanSingleIC"] = float(np.mean(singles))
        row["BestSingle"] = config.MODELS[int(np.argmax(singles))]
        row["EnsembleBeatsMeanSingle"] = bool(
            row["IC_ensemble_stacked"] > row["MeanSingleIC"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def run(paths=config.PATHS, assets=None, n_jobs=-1):
    forecasts = run_all(paths, assets, n_jobs)
    table = evaluate(forecasts)
    table.to_csv(paths.tables / "model_comparison.csv", index=False)
    return forecasts, table

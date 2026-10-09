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
CONVEX_PENALTY = 1e3
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


def walk_forward_lstm(
    X, y, seed=config.SEED, start=None, hidden=None, lr=None, l2=None, scale_target=True
):
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


def stack_forecasts(
    P, y, window=None, fallback=None, return_weights=False, convex=False
):
    window = config.STACK_WINDOW if window is None else window
    base = P.mean(axis=1, skipna=False) if fallback is None else fallback
    stacked = base.copy()
    values, target = P.values, y.values
    weights = pd.DataFrame(np.nan, index=P.index, columns=P.columns)
    for i in range(len(P)):
        if np.isnan(values[i]).any():
            continue
        hist = np.arange(i)
        ok = hist[~np.isnan(values[hist]).any(axis=1) & ~np.isnan(target[hist])]
        if len(ok) < window:
            continue
        ok = ok[-window:]
        if convex:
            lam = CONVEX_PENALTY
            A = np.vstack([values[ok], np.full((1, values.shape[1]), lam)])
            w, _ = nnls(A, np.append(target[ok], lam))
            w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1.0 / len(w))
        else:
            w, _ = nnls(values[ok], target[ok])
        stacked.iloc[i] = values[i] @ w
        weights.iloc[i] = w
    equal = P.mean(axis=1, skipna=False)
    return (equal, stacked, weights) if return_weights else (equal, stacked)


def calibrate_to_mean(pred, y, base, window=None, min_obs=None):
    window = config.CALIBRATION_WINDOW if window is None else window
    min_obs = config.CALIBRATION_MIN if min_obs is None else min_obs
    p, t, b = pred.values, y.values, base.values
    out, slopes = base.copy(), pd.Series(np.nan, index=pred.index)
    for i in range(len(p)):
        if np.isnan(p[i]) or np.isnan(b[i]):
            continue
        hist = np.arange(i)
        ok = hist[~np.isnan(p[hist]) & ~np.isnan(t[hist]) & ~np.isnan(b[hist])][
            -window:
        ]
        slope = 0.0
        if len(ok) >= min_obs:
            d, e = p[ok] - b[ok], t[ok] - b[ok]
            den = float(d @ d)
            slope = float(np.clip((d @ e) / den, 0.0, 1.0)) if den > 0 else 0.0
        slopes.iloc[i] = slope
        out.iloc[i] = b[i] + slope * (p[i] - b[i])
    return out, slopes


def run_asset(asset, F, log_price, seed=config.SEED, pooled=None):
    rows = asset_frame(F, asset)
    y = forward_targets(log_price.reindex(rows.index)).values
    cols = feature_columns(F, asset)
    X = rows[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    preds = walk_forward_sklearn(X, y, ["xgb", "rf", "ridge"], seed)
    lstm_cols = [f"{asset}_{c}" for c in LSTM_FEATURES] + ["VIX_zscore_252d", "regime"]
    X_lstm = rows[lstm_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    preds["lstm"] = walk_forward_lstm(X_lstm, y, seed)
    out = pd.DataFrame({"target": y, **preds}, index=rows.index)
    if pooled is not None:
        for name in config.POOLED:
            out[name] = pooled[name].reindex(out.index)
    out["hist_mean"] = historical_mean(out["target"])
    equal, stacked = stack_forecasts(out[config.MODELS], out["target"])
    out["ensemble_equal"], out["ensemble_stacked"] = equal, stacked
    cand = [c for c in config.CANDIDATES if c in out]
    _, stack_all, weights = stack_forecasts(
        out[cand],
        out["target"],
        fallback=out["hist_mean"],
        return_weights=True,
        convex=True,
    )
    out["stack_all"] = stack_all
    out["ensemble_final"], out["calib_slope"] = calibrate_to_mean(
        stack_all, out["target"], out["hist_mean"]
    )
    for c in cand:
        out[f"w_{c}"] = weights[c]
    gated = ["stack_all", "ensemble_final", "calib_slope"]
    out.loc[out.index[: config.INITIAL_TRAIN_MONTHS], gated] = np.nan
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
    from models.pooled import pooled_forecasts

    pooled = pooled_forecasts(F, log_price, chosen, seed)
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_asset)(a, F, log_price[a], seed, pooled[a]) for a in chosen
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


def historical_mean(target):
    return target.expanding().mean().shift(1)


def r2_oos(pred, target, benchmark):
    denom = float(np.sum((target - benchmark) ** 2))
    return 1.0 - float(np.sum((target - pred) ** 2)) / denom if denom > 0 else np.nan


SKILL_COLUMNS = (
    config.MODELS
    + config.POOLED
    + ["ensemble_equal", "ensemble_stacked", "stack_all", "ensemble_final"]
)


def evaluate(forecasts):
    rows = []
    for asset, df in forecasts.items():
        cols = [c for c in SKILL_COLUMNS if c in df]
        oos = df.dropna(subset=["target"] + cols)
        row = {"Asset": asset, "n_oos": len(oos)}
        if len(oos) < 12:
            rows.append(row)
            continue
        base = float((oos["target"] > 0).mean())
        bench = historical_mean(df["target"]).reindex(oos.index)
        row["BaseRate"] = base
        row["UpShare_stacked"] = float((oos["ensemble_stacked"] > 0).mean())
        for c in cols:
            ic, p = spearmanr(oos[c], oos["target"])
            row[f"IC_{c}"] = ic
            hit = float((np.sign(oos[c]) == np.sign(oos["target"])).mean())
            row[f"Hit_{c}"] = hit
            row[f"HitEdge_{c}"] = hit - base
            row[f"R2OOS_{c}"] = r2_oos(oos[c], oos["target"], bench)
            row[f"Disp_{c}"] = float(oos[c].std() / oos["target"].std())
            if c == "ensemble_stacked":
                row["p_value_stacked"] = p
            if c == "ensemble_final":
                row["p_value_final"] = p
        singles = [row[f"IC_{m}"] for m in config.MODELS]
        row["MeanSingleIC"] = float(np.mean(singles))
        row["BestSingle"] = config.MODELS[int(np.argmax(singles))]
        row["EnsembleBeatsMeanSingle"] = bool(
            row["IC_ensemble_stacked"] > row["MeanSingleIC"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def skill_by_period(forecasts, cut=None):
    cut = pd.Timestamp(config.HOLDOUT_START if cut is None else cut)
    periods = (
        ("full", lambda i: np.ones(len(i), dtype=bool)),
        ("development", lambda i: i < cut),
        ("holdout", lambda i: i >= cut),
    )
    rows = []
    for period, pick in periods:
        for c in SKILL_COLUMNS:
            ics, r2s = [], []
            for df in forecasts.values():
                if c not in df:
                    continue
                cols = [k for k in SKILL_COLUMNS if k in df]
                oos = df.dropna(subset=["target"] + cols)
                oos = oos[pick(oos.index)]
                if len(oos) < 12:
                    continue
                bench = historical_mean(df["target"]).reindex(oos.index)
                ics.append(spearmanr(oos[c], oos["target"])[0])
                r2s.append(r2_oos(oos[c], oos["target"], bench))
            if ics:
                rows.append(
                    {
                        "Period": period,
                        "Forecast": c,
                        "MeanIC": float(np.nanmean(ics)),
                        "MeanR2OOS": float(np.nanmean(r2s)),
                        "AssetsPositiveR2": int(np.sum(np.array(r2s) > 0)),
                        "Assets": len(r2s),
                    }
                )
    return pd.DataFrame(rows)


def run(paths=config.PATHS, assets=None, n_jobs=-1):
    forecasts = run_all(paths, assets, n_jobs)
    table = evaluate(forecasts)
    table.to_csv(paths.tables / "model_comparison.csv", index=False)
    skill_by_period(forecasts).to_csv(paths.tables / "skill_by_period.csv", index=False)
    weights = pd.DataFrame(
        {
            a: df[[c for c in df.columns if c.startswith("w_")]].mean()
            for a, df in forecasts.items()
        }
    ).T
    weights.columns = [c[2:] for c in weights.columns]
    weights.index.name = "Asset"
    weights.to_csv(paths.tables / "stack_weights.csv")
    return forecasts, table

import hashlib
import json

import pandas as pd

import config
from backtest.engine import performance

SPEC_DIRS = ("models", "features")
SPEC_FILES = ("backtest/engine.py", "backtest/strategies.py", "backtest/stats.py")


def spec_hash(paths=config.PATHS):
    consts = {
        k: v
        for k, v in vars(config).items()
        if k.isupper() and isinstance(v, (int, float, str, list, dict, tuple))
    }
    h = hashlib.sha256(json.dumps(consts, sort_keys=True, default=str).encode())
    files = [f for d in SPEC_DIRS for f in sorted((paths.root / d).glob("*.py"))]
    files += [paths.root / f for f in SPEC_FILES]
    for f in files:
        if f.exists():
            h.update(f.read_bytes())
    return h.hexdigest()


def freeze(paths=config.PATHS):
    path = paths.root / "results" / "freeze.json"
    if path.exists():
        return json.loads(path.read_text())
    ret = pd.read_csv(
        paths.processed / "backtest_returns.csv", index_col=0, parse_dates=True
    )
    record = {
        "freeze_date": str(ret.index[-1].date()),
        "spec_hash": spec_hash(paths),
        "months_at_freeze": int(len(ret)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))
    return record


def evaluate(paths=config.PATHS):
    record = freeze(paths)
    ret = pd.read_csv(
        paths.processed / "backtest_returns.csv", index_col=0, parse_dates=True
    )
    rf = pd.read_csv(
        paths.processed / "backtest_rf.csv", index_col=0, parse_dates=True
    )["rf"]
    new = ret[ret.index > pd.Timestamp(record["freeze_date"])]
    unchanged = spec_hash(paths) == record["spec_hash"]
    status = pd.DataFrame(
        [
            {
                "freeze_date": record["freeze_date"],
                "spec_hash": record["spec_hash"][:16],
                "spec_unchanged": unchanged,
                "months_since_freeze": len(new),
                "inference_possible": bool(unchanged and len(new) >= 24),
            }
        ]
    )
    status.to_csv(paths.tables / "holdout_status.csv", index=False)
    if len(new):
        rows = {}
        for n in new.columns:
            bt = pd.DataFrame(
                {"net": new[n], "rf": rf.reindex(new.index), "turnover": 0.0}
            )
            rows[n] = {k: v for k, v in performance(bt).items() if k != "AnnTurnover"}
        pd.DataFrame(rows).T.to_csv(paths.tables / "holdout_metrics.csv")
    else:
        (paths.tables / "holdout_metrics.csv").unlink(missing_ok=True)
    return status


def run(paths=config.PATHS):
    return evaluate(paths)

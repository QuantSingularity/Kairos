import argparse
import time
from importlib import import_module

from kairos import config

STAGE_MODULES = {
    "data": "kairos.data.loader",
    "regimes": "kairos.models.regime_detection",
    "features": "kairos.features.build_features",
    "forecast": "kairos.models.forecasting",
    "diagnostics": "kairos.models.diagnostics",
    "shap": "kairos.models.shap_analysis",
    "portfolio": "kairos.backtest.engine",
    "sensitivity": "kairos.backtest.sensitivity",
    "windows": "kairos.backtest.windows",
    "holdout": "kairos.backtest.holdout",
    "figures": "kairos.reporting.figures",
    "report": "kairos.reporting.report",
    "deck": "kairos.reporting.deck",
    "notebooks": "kairos.reporting.notebooks",
}
STAGES = list(STAGE_MODULES)


def run_stage(name, paths=config.PATHS, refresh=False, assets=None, n_jobs=-1):
    if name not in STAGE_MODULES:
        raise ValueError(f"unknown stage {name}")
    module = import_module(STAGE_MODULES[name])
    if name == "data":
        return module.run(paths, refresh)
    if name in ("forecast", "diagnostics"):
        return module.run(paths, assets, n_jobs)
    return module.run(paths)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Kairos: regime-aware portfolio intelligence pipeline",
    )
    parser.add_argument("stage", choices=STAGES + ["all"])
    parser.add_argument(
        "--refresh", action="store_true", help="re-download prices from Yahoo Finance"
    )
    parser.add_argument(
        "--assets", nargs="*", help="restrict forecasting to these tickers"
    )
    parser.add_argument(
        "--jobs", type=int, default=-1, help="parallel workers for forecasting"
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config.PATHS.ensure()
    for stage in STAGES if args.stage == "all" else [args.stage]:
        t0 = time.time()
        run_stage(stage, config.PATHS, args.refresh, args.assets, args.jobs)
        print(f"{stage:<10} done in {time.time() - t0:6.1f}s")

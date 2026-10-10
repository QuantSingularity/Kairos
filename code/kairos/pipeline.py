"""Stage registry and command line logic."""

import argparse
import logging
import time
from importlib import import_module
from typing import Any

from kairos import config, logs

log = logging.getLogger(__name__)

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


def run_stage(
    name: str,
    paths: config.Paths = config.PATHS,
    refresh: bool = False,
    assets: list[str] | None = None,
    n_jobs: int = -1,
) -> Any:
    """Run one stage by name."""
    if name not in STAGE_MODULES:
        raise ValueError(f"unknown stage {name}")
    module = import_module(STAGE_MODULES[name])
    if name == "data":
        return module.run(paths, refresh)
    if name in ("forecast", "diagnostics"):
        return module.run(paths, assets, n_jobs)
    return module.run(paths)


def build_parser() -> argparse.ArgumentParser:
    """Command line parser."""
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
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="logging verbosity",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Command line entry point."""
    args = build_parser().parse_args(argv)
    logs.configure(args.log_level)
    config.PATHS.ensure()
    for stage in STAGES if args.stage == "all" else [args.stage]:
        t0 = time.time()
        run_stage(stage, config.PATHS, args.refresh, args.assets, args.jobs)
        log.info("stage %s finished in %.1fs", stage, time.time() - t0)

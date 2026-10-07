import argparse
import time

import config

STAGES = [
    "data",
    "regimes",
    "features",
    "forecast",
    "shap",
    "portfolio",
    "figures",
    "report",
    "deck",
    "notebooks",
]


def run_stage(name, paths=config.PATHS, refresh=False, assets=None, n_jobs=-1):
    if name == "data":
        from data import loader

        return loader.run(paths, refresh)
    if name == "regimes":
        from models import regime_detection

        return regime_detection.run(paths)
    if name == "features":
        from features import build_features

        return build_features.run(paths)
    if name == "forecast":
        from models import forecasting

        return forecasting.run(paths, assets, n_jobs)
    if name == "shap":
        from models import shap_analysis

        return shap_analysis.run(paths)
    if name == "portfolio":
        from backtest import engine

        return engine.run(paths)
    if name == "figures":
        from reporting import figures

        return figures.run(paths)
    if name == "report":
        from reporting import report

        return report.run(paths)
    if name == "deck":
        from reporting import deck

        return deck.run(paths)
    if name == "notebooks":
        from reporting import notebooks

        return notebooks.run(paths)
    raise ValueError(f"unknown stage {name}")


def main():
    parser = argparse.ArgumentParser(
        description="Kairos: regime-aware portfolio intelligence pipeline"
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
    args = parser.parse_args()
    config.PATHS.ensure()
    for stage in STAGES if args.stage == "all" else [args.stage]:
        t0 = time.time()
        run_stage(stage, config.PATHS, args.refresh, args.assets, args.jobs)
        print(f"{stage:<10} done in {time.time() - t0:6.1f}s")


if __name__ == "__main__":
    main()

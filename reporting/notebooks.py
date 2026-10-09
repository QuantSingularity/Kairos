import nbformat as nbf

import config

HEADER = "import sys\nsys.path.insert(0, '..')\nimport pandas as pd\nimport config\nP = config.PATHS\n"

SPECS = {
    "01_data_and_regimes": [
        (
            "md",
            "# Data and regimes\nData quality report, regime validation, and the regime timeline.",
        ),
        ("code", HEADER + "pd.read_csv(P.processed / 'data_quality.csv')"),
        ("code", "pd.read_csv(P.tables / 'regime_validation.csv')"),
        (
            "code",
            "from IPython.display import Image\nImage(str(P.figures / '02_regime_timeline.png'))",
        ),
    ],
    "02_forecasting": [
        (
            "md",
            "# Forecasting\nOut-of-sample information coefficients by asset and model.",
        ),
        ("code", HEADER + "pd.read_csv(P.tables / 'model_comparison.csv').T"),
        (
            "code",
            "from IPython.display import Image\nImage(str(P.figures / '04_model_ic.png'))",
        ),
        (
            "code",
            "from IPython.display import Image\nImage(str(P.figures / '11_forecast_skill.png'))",
        ),
        ("code", "pd.read_csv(P.tables / 'shap_SPY.csv', index_col=0).head(15)"),
    ],
    "04_diagnostics_and_robustness": [
        (
            "md",
            "# Diagnostics and robustness\nForecast diagnostics, overlay ablation, deflated Sharpe and parameter sensitivity.",
        ),
        (
            "code",
            HEADER + "pd.read_csv(P.tables / 'regime_return_link.csv')",
        ),
        ("code", "pd.read_csv(P.tables / 'ic_diagnostics_summary.csv', index_col=0)"),
        (
            "code",
            "from IPython.display import Image\nImage(str(P.figures / '12_ic_diagnostics.png'))",
        ),
        ("code", "pd.read_csv(P.tables / 'deflated_sharpe.csv')"),
        (
            "code",
            "from IPython.display import Image\nImage(str(P.figures / '14_overlay_ablation.png'))",
        ),
        ("code", "pd.read_csv(P.tables / 'lstm_repair.csv')"),
        ("code", "pd.read_csv(P.tables / 'skill_by_period.csv')"),
        (
            "code",
            "from IPython.display import Image\nImage(str(P.figures / '15_forecast_repair.png'))",
        ),
        ("code", "pd.read_csv(P.tables / 'period_split.csv')"),
        ("code", "pd.read_csv(P.tables / 'window_robustness.csv')"),
        (
            "code",
            "from IPython.display import Image\nImage(str(P.figures / '16_window_robustness.png'))",
        ),
        ("code", "pd.read_csv(P.tables / 'prior_comparison.csv')"),
        (
            "code",
            "from IPython.display import Image\nImage(str(P.figures / '17_prior_comparison.png'))",
        ),
        ("code", "pd.read_csv(P.tables / 'holdout_status.csv')"),
        ("code", "pd.read_csv(P.tables / 'sensitivity.csv')"),
        (
            "code",
            "from IPython.display import Image\nImage(str(P.figures / '13_sensitivity.png'))",
        ),
    ],
    "03_portfolio_and_backtest": [
        ("md", "# Portfolio construction and backtest"),
        ("code", HEADER + "pd.read_csv(P.tables / 'summary_metrics.csv', index_col=0)"),
        ("code", "pd.read_csv(P.tables / 'significance.csv')"),
        (
            "code",
            "from IPython.display import Image\nImage(str(P.figures / '07_cumulative_returns.png'))",
        ),
        ("code", "pd.read_csv(P.tables / 'regime_metrics.csv', index_col=[0, 1])"),
        ("code", "pd.read_csv(P.tables / 'cost_sensitivity.csv', index_col=0)"),
    ],
}


def execute(path):
    if not (path.parent.parent / "config.py").exists():
        return False
    try:
        from nbclient import NotebookClient
    except ImportError:
        return False
    nb = nbf.read(path, as_version=4)
    NotebookClient(
        nb,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    ).execute()
    nbf.write(nb, path)
    return True


def run(paths=config.PATHS):
    paths.notebooks.mkdir(parents=True, exist_ok=True)
    out = []
    for name, cells in SPECS.items():
        nb = nbf.v4.new_notebook()
        nb.cells = [
            nbf.v4.new_markdown_cell(t) if kind == "md" else nbf.v4.new_code_cell(t)
            for kind, t in cells
        ]
        path = paths.notebooks / f"{name}.ipynb"
        nbf.write(nb, path)
        execute(path)
        out.append(path)
    return out

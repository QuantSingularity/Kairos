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
        ("code", HEADER + "pd.read_csv(P.tables / 'model_comparison.csv')"),
        (
            "code",
            "from IPython.display import Image\nImage(str(P.figures / '04_model_ic.png'))",
        ),
        ("code", "pd.read_csv(P.tables / 'shap_SPY.csv', index_col=0).head(15)"),
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
        out.append(path)
    return out

import numpy as np
import pandas as pd
import pytest
from fixtures import make_panel, write_raw_cache

import config
import run

ASSETS = ["SPY", "EFA", "TLT", "IEF", "GLD"]


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    paths = config.Paths(tmp_path_factory.mktemp("proj"))
    paths.ensure()
    panel = make_panel(start="2004-01-01", end="2017-12-31")
    others = [c for c in config.ASSETS if c not in ASSETS]
    write_raw_cache(paths, panel.assign(**{c: np.nan for c in others}))
    return paths


@pytest.fixture(scope="module")
def staged(project):
    mp = pytest.MonkeyPatch()
    mp.setattr(config, "INITIAL_TRAIN_MONTHS", 36)
    mp.setattr(config, "LSTM_EPOCHS", 3)
    mp.setattr(config, "LSTM_REFIT_EVERY", 24)
    mp.setattr(config, "BOOTSTRAP_DRAWS", 150)
    mp.setattr(config, "SHAP_ASSETS", ["SPY"])
    mp.setattr(config, "DIAG_INITIAL_TRAIN", 48)
    mp.setattr(config, "DIAG_MODELS", ["rf", "ridge"])
    mp.setattr(
        config,
        "SENSITIVITY_GRID",
        {"tau": [0.05, 0.1], "prior": ["reference", "equal"]},
    )
    for stage in ("data", "regimes", "features"):
        run.run_stage(stage, project)
    run.run_stage("forecast", project, assets=ASSETS, n_jobs=1)
    run.run_stage("diagnostics", project, assets=ASSETS, n_jobs=1)
    yield project
    mp.undo()


def test_stage_outputs(staged):
    for f in (
        "panel_daily.csv",
        "regimes_daily.csv",
        "features_monthly.csv",
        "forecasts_SPY.csv",
        "data_quality.csv",
    ):
        assert (staged.processed / f).stat().st_size > 0
    f = pd.read_csv(
        staged.processed / "forecasts_SPY.csv", index_col=0, parse_dates=True
    )
    assert f["ensemble_stacked"].notna().sum() > 20


def test_full_chain_to_reports(staged):
    run.run_stage("shap", staged)
    out = run.run_stage("portfolio", staged)
    assert {
        "BL_ML_Regime",
        "BL_ML_NoRegime",
        "BL_HistMean_Regime",
        "Equal_Weight",
        "MeanVariance_LW",
        "Regime_60_40",
        "Regime_Market",
    } <= set(out["summary"].index)
    assert set(out["deflated"]["Trials"]) == set(config.DSR_TRIALS)
    sens = run.run_stage("sensitivity", staged)
    assert len(sens) == 4 * 2 and sens["IsDefault"].sum() == 4
    assert np.isfinite(out["summary"]["Sharpe"]).all()
    assert len({len(bt) for bt in out["returns"].values()}) == 1
    for w in out["weights"].values():
        assert np.allclose(w.sum(axis=1), 1.0)
    run.run_stage("figures", staged)
    report = run.run_stage("report", staged)
    deck = run.run_stage("deck", staged)
    run.run_stage("notebooks", staged)
    text = report.read_text(encoding="utf-8")
    assert "Research Question" in text and "p = " in text
    assert (
        "Overlay ablation" in text and "Diagnosing the information coefficient" in text
    )
    assert "significantly negative" in text
    assert "\u2014" not in text and "\u2013" not in text
    assert deck.exists() and len(list(staged.figures.glob("*.png"))) == 14
    import shutil

    if shutil.which("pandoc"):
        assert (staged.reports / "Kairos_Report.docx").exists()


def test_no_look_ahead_in_regime_feature(staged):
    F = pd.read_csv(
        staged.processed / "features_monthly.csv", index_col=0, parse_dates=True
    )
    R = pd.read_csv(
        staged.processed / "regimes_daily.csv", index_col=0, parse_dates=True
    )
    d = F["regime"].dropna().index[10]
    assert F.at[d, "regime"] == R.at[d, "regime"]

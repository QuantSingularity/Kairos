import numpy as np
import pandas as pd
import pytest
from kairos import config, pipeline
from tests.fixtures import make_panel, write_raw_cache

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
    mp.setattr(config, "CRISIS_INITIAL_TRAIN", 24)
    mp.setattr(config, "HOLDOUT_START", "2013-01-01")
    mp.setattr(config, "DIAG_MODELS", ["rf", "ridge"])
    mp.setattr(
        config,
        "SENSITIVITY_GRID",
        {"tau": [0.05, 0.1], "prior": ["reference", "equal"]},
    )
    for stage in ("data", "regimes", "features"):
        pipeline.run_stage(stage, project)
    pipeline.run_stage("forecast", project, assets=ASSETS, n_jobs=1)
    pipeline.run_stage("diagnostics", project, assets=ASSETS, n_jobs=1)
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
    assert {"pooled_ridge", "hist_mean", "stack_all", "ensemble_final"} <= set(
        f.columns
    )
    first = f.index[config.INITIAL_TRAIN_MONTHS]
    assert f["ensemble_final"].first_valid_index() == first
    assert f["stack_all"].first_valid_index() == first
    assert f["ensemble_stacked"].first_valid_index() >= first
    assert (staged.processed / "data_manifest.csv").stat().st_size > 0


def test_full_chain_to_reports(staged):
    pipeline.run_stage("shap", staged)
    out = pipeline.run_stage("portfolio", staged)
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
    sens = pipeline.run_stage("sensitivity", staged)
    assert len(sens) == 4 * 2 and sens["IsDefault"].sum() == 4
    priors = pd.read_csv(staged.tables / "prior_comparison.csv")
    assert set(priors["Prior"]) == {"reference", "equal", "inverse_vol"}
    assert (staged.tables / "period_split.csv").exists()
    pipeline.run_stage("windows", staged)
    win = pd.read_csv(staged.tables / "window_robustness.csv", index_col=0)
    meta = pd.read_csv(staged.tables / "window_meta.csv")
    assert win["Months_crisis_window"].iloc[0] > win["Months_main"].iloc[0]
    assert meta["first_decision"].iloc[0] < out["returns"]["BL_ML_Regime"].index[
        0
    ].strftime("%Y-%m-%d")
    status = pipeline.run_stage("holdout", staged).iloc[0]
    assert status["months_since_freeze"] == 0 and status["spec_unchanged"]
    assert (staged.root / "results" / "freeze.json").exists()
    assert np.isfinite(out["summary"]["Sharpe"]).all()
    assert len({len(bt) for bt in out["returns"].values()}) == 1
    for w in out["weights"].values():
        assert np.allclose(w.sum(axis=1), 1.0)
    pipeline.run_stage("figures", staged)
    report = pipeline.run_stage("report", staged)
    deck = pipeline.run_stage("deck", staged)
    pipeline.run_stage("notebooks", staged)
    text = report.read_text(encoding="utf-8")
    assert "Research Question" in text and "p = " in text
    assert (
        "Overlay ablation" in text and "Diagnosing the information coefficient" in text
    )
    assert "Repairing the forecasts" in text and "Frozen specification" in text
    assert "Crisis-inclusive window" in text and "prior portfolio" in text
    assert "significantly negative" in text
    assert "\u2014" not in text and "\u2013" not in text
    assert deck.exists() and len(list(staged.figures.glob("*.png"))) == 17
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

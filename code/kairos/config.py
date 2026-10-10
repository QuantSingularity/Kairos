import os
from dataclasses import dataclass
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("KAIROS_ROOT", PACKAGE_DIR.parents[1])).resolve()


@dataclass(frozen=True)
class Paths:
    root: Path = ROOT

    @property
    def raw(self):
        return self.root / "data" / "raw"

    @property
    def processed(self):
        return self.root / "data" / "processed"

    @property
    def results(self):
        return self.root / "results"

    @property
    def tables(self):
        return self.results / "tables"

    @property
    def figures(self):
        return self.results / "figures"

    @property
    def reports(self):
        return self.root / "reports"

    @property
    def notebooks(self):
        return self.root / "notebooks"

    def ensure(self):
        for p in (
            self.raw,
            self.processed,
            self.tables,
            self.figures,
            self.reports,
            self.notebooks,
        ):
            p.mkdir(parents=True, exist_ok=True)


PATHS = Paths()

ASSETS = ["SPY", "EFA", "TLT", "IEF", "HYG", "GLD", "DBC", "BTC-USD", "ETH-USD"]
CRYPTO = ["BTC-USD", "ETH-USD"]
REQUIRED = ["SPY", "EFA", "TLT", "IEF", "VIX", "RF"]
VIX_TICKER = "^VIX"
RF_TICKER = "^IRX"
START_DATE = "2003-01-01"
END_DATE = "2026-09-30"
SEED = 690

HMM_MIN_TRAIN = 504
HMM_REFIT_EVERY = 252
STRESS_THRESHOLD = 0.5

INITIAL_TRAIN_MONTHS = 48
LSTM_SEQ_LEN = 12
LSTM_REFIT_EVERY = 6
LSTM_EPOCHS = 40
LSTM_PATIENCE = 6
LSTM_HIDDEN = 12
LSTM_LR = 0.01
LSTM_L2 = 1e-4
STACK_WINDOW = 24
MODELS = ["xgb", "rf", "ridge", "lstm"]
POOLED = ["pooled_ridge", "pooled_xgb"]
CANDIDATES = MODELS + POOLED + ["hist_mean"]
CALIBRATION_WINDOW = 36
CALIBRATION_MIN = 24
VIEW_COLUMN = "ensemble_final"
RAW_VIEW_COLUMN = "ensemble_stacked"
HOLDOUT_START = "2019-01-01"
CRISIS_INITIAL_TRAIN = 24
OBSERVED_PRIOR_FILE = "prior_weights.csv"

MARKET_WEIGHTS = {
    "SPY": 0.42,
    "EFA": 0.12,
    "TLT": 0.10,
    "IEF": 0.15,
    "GLD": 0.08,
    "DBC": 0.05,
    "HYG": 0.05,
    "BTC-USD": 0.02,
    "ETH-USD": 0.01,
}
TAU = 0.05
BASE_DELTA = 2.5
STRESSED_DELTA_MULT = 1.8
STRESSED_OMEGA_MULT = 2.5
IC_FLOOR = 0.05
MAX_WEIGHT = 0.35
COV_WINDOW = 60
MIN_HISTORY = 24
MIN_UNIVERSE = 3

COST_BPS = 10
COST_GRID = [0, 10, 25, 50]
BOOTSTRAP_DRAWS = 4000
BOOTSTRAP_BLOCK = 6
SHAP_ASSETS = ["SPY", "EFA", "TLT", "GLD"]

KAIROS = ["BL_ML_Regime", "BL_ML_Regime_IC", "BL_ML_NoRegime"]
CLASSICAL = [
    "Market_Reference",
    "Equal_Weight",
    "Static_60_40",
    "Inverse_Vol",
    "MeanVariance_LW",
]
OVERLAYS = ["Regime_60_40", "Regime_Market"]
NAIVE = "BL_HistMean_Regime"
RAW = "BL_ML_Raw_Regime"
ABLATIONS = ["BL_ML_Raw_Regime", "BL_HistMean_Regime"]
DSR_TRIALS = (5, 10)

DIAG_INITIAL_TRAIN = 72
DIAG_MODELS = ["xgb", "rf", "ridge"]

SENSITIVITY_GRID = {
    "tau": [0.025, 0.05, 0.1, 0.2],
    "base_delta": [1.5, 2.5, 4.0, 6.0],
    "stressed_delta_mult": [1.0, 1.4, 1.8, 2.5],
    "stressed_omega_mult": [1.0, 1.5, 2.5, 4.0],
    "max_weight": [0.25, 0.35, 0.5],
    "prior": ["reference", "equal", "inverse_vol"],
}

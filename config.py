from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent


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
    def tables(self):
        return self.root / "results" / "tables"

    @property
    def figures(self):
        return self.root / "results" / "figures"

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
STACK_WINDOW = 24
MODELS = ["xgb", "rf", "ridge", "lstm"]

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

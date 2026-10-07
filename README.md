# Kairos: Regime-Aware Portfolio Intelligence

Kairos is a regime-aware machine learning approach to portfolio allocation. It asks whether an allocation engine that detects market regimes, forecasts asset returns with an ensemble of models, and turns those forecasts into Black-Litterman views can improve risk-adjusted returns over classical allocation methods, and under which market conditions its advantage holds.

|                   |                                                                             |
| ----------------- | --------------------------------------------------------------------------- |
| **Language**      | Python 3.11                                                                 |
| **Universe**      | 9 assets across equities, bonds, credit, gold, commodities and crypto       |
| **Regime engine** | Walk-forward two-state Gaussian HMM                                         |
| **Forecasters**   | XGBoost, Random Forest, Ridge, NumPy LSTM, equal and stacked ensembles      |
| **Allocator**     | Black-Litterman with Ledoit-Wolf covariance and a cash asset                |
| **Evaluation**    | Cost-aware monthly backtest, paired bootstrap, regime and sub-period splits |

---

## Quick start

```
pip install -r requirements.txt
python run.py all
```

The first run downloads prices from Yahoo Finance and caches them in `data/raw/`. Later runs reuse the cache. You can also place your own `Date,Close` CSV files there, named like `SPY.csv`, `BTC_USD.csv`, `VIX.csv` and `IRX.csv`. If Pandoc is installed, the report is also converted to Word.

| Command                 | Purpose                                          |
| ----------------------- | ------------------------------------------------ |
| `python run.py all`     | Run every stage in order                         |
| `python run.py <stage>` | Run a single stage                               |
| `--refresh`             | Download prices again instead of using the cache |
| `--assets SPY TLT`      | Restrict forecasting to the listed tickers       |
| `--jobs N`              | Number of parallel workers for forecasting       |

---

## Pipeline

| #   | Stage       | What it does                                        | Main output                            |
| --- | ----------- | --------------------------------------------------- | -------------------------------------- |
| 1   | `data`      | Downloads, aligns and validates the price panel     | `data/processed/panel_daily.csv`       |
| 2   | `regimes`   | Fits the walk-forward HMM and the volatility rule   | `data/processed/regimes_daily.csv`     |
| 3   | `features`  | Builds causal monthly features                      | `data/processed/features_monthly.csv`  |
| 4   | `forecast`  | Runs walk-forward forecasts for every asset         | `data/processed/forecasts_<asset>.csv` |
| 5   | `shap`      | Explains the forecasts by regime                    | `results/tables/shap_<asset>.csv`      |
| 6   | `portfolio` | Builds portfolios, backtests and tests significance | `results/tables/*.csv`                 |
| 7   | `figures`   | Draws the ten report figures                        | `results/figures/*.png`                |
| 8   | `report`    | Writes the report in Markdown and Word              | `reports/Kairos_Report.md`             |
| 9   | `deck`      | Builds the slide deck                               | `reports/Kairos_Deck.pptx`             |
| 10  | `notebooks` | Writes three results notebooks                      | `notebooks/*.ipynb`                    |

---

## Data

Daily adjusted closes come from Yahoo Finance between 2003 and September 2026. Nothing is simulated.

| Ticker  | Role                                       | Reference weight |
| ------- | ------------------------------------------ | ---------------- |
| SPY     | US large-cap equities                      | 42%              |
| EFA     | International developed equities           | 12%              |
| IEF     | Intermediate US Treasuries                 | 15%              |
| TLT     | Long US Treasuries                         | 10%              |
| GLD     | Gold                                       | 8%               |
| DBC     | Broad commodities                          | 5%               |
| HYG     | High-yield credit                          | 5%               |
| BTC-USD | Bitcoin                                    | 2%               |
| ETH-USD | Ethereum                                   | 1%               |
| ^VIX    | Volatility input for regimes and features  | n/a              |
| ^IRX    | 13-week T-bill, used as the risk-free rate | n/a              |

`validate_panel` checks every run before any model sees the data.

| Check                   | Purpose                                                                      |
| ----------------------- | ---------------------------------------------------------------------------- |
| History length          | Required series must cover at least 8 years                                  |
| Gaps                    | No more than 2% missing values after a series starts                         |
| Daily moves             | Rejects moves above 60% (80% for crypto) as data errors                      |
| Cross-asset correlation | Floors on SPY/EFA, TLT/IEF and SPY/HYG detect substitute or corrupted series |

---

## Method

### Regimes

| Item        | Setting                                            |
| ----------- | -------------------------------------------------- |
| Model       | Two-state Gaussian HMM                             |
| Inputs      | SPY daily return, 21-day realised volatility, VIX  |
| Refit       | Once a year on past data only                      |
| Daily state | Forward filter, so no label uses later information |
| Stress flag | Filtered stress probability above 0.5              |
| Comparison  | Rolling 80th-percentile volatility rule            |

### Forecasting

| Item          | Setting                                                                                                                                           |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| Target        | Next-month log return per asset                                                                                                                   |
| Features      | About 20 causal technical, volatility, momentum, drawdown and beta features per asset, plus VIX level, change and z-score, and the regime signals |
| Single models | XGBoost, Random Forest, Ridge, NumPy LSTM                                                                                                         |
| LSTM          | One layer, 12 hidden units, 12-month sequences, refit every 6 months                                                                              |
| Ensembles     | Equal weight, and a stacked non-negative least squares blend over the trailing 24 months                                                          |
| Validation    | Expanding walk-forward after a 48-month initial training window                                                                                   |
| Skill metrics | Information coefficient and hit rate by asset and model                                                                                           |

### Portfolio construction

| Item             | Setting                                                          |
| ---------------- | ---------------------------------------------------------------- |
| Prior            | Reference market weights, scaled to the eligible universe        |
| Views            | Stacked forecasts as absolute views                              |
| View uncertainty | Trailing forecast error variance                                 |
| Covariance       | Ledoit-Wolf shrinkage on a 60-month window                       |
| Cash             | Optimiser may hold cash, so higher risk aversion lowers exposure |
| Position cap     | 35% per asset                                                    |
| Stress response  | Risk aversion multiplied by 1.8 and view uncertainty by 2.5      |

| Variant           | Description                                                  |
| ----------------- | ------------------------------------------------------------ |
| `BL_ML_Regime`    | Black-Litterman with regime scaling                          |
| `BL_ML_Regime_IC` | Regime scaling plus views scaled by each asset's trailing IC |
| `BL_ML_NoRegime`  | Same engine without regime scaling, used as the ablation     |

### Benchmarks

| Benchmark          | Description                                            |
| ------------------ | ------------------------------------------------------ |
| `Market_Reference` | Reference market weights                               |
| `Equal_Weight`     | Equal weight across the eligible universe              |
| `Static_60_40`     | 60% SPY and 40% IEF                                    |
| `Inverse_Vol`      | Weights proportional to inverse 36-month volatility    |
| `MeanVariance_LW`  | Mean-variance optimisation with Ledoit-Wolf covariance |

All benchmarks use the same eligible universe as the Kairos portfolios.

---

## Evaluation

| Component         | Detail                                                                                                     |
| ----------------- | ---------------------------------------------------------------------------------------------------------- |
| Rebalancing       | Monthly, with weight drift between rebalances                                                              |
| Transaction costs | 10 bps on traded notional, with a 0, 10, 25 and 50 bps grid                                                |
| Cash return       | T-bill rate                                                                                                |
| Sharpe ratio      | Measured on excess returns over the T-bill rate                                                            |
| Significance      | Paired stationary bootstrap of the Sharpe difference, centred on the null, 4,000 draws with 6-month blocks |
| Regime split      | Calm and stressed decision months reported separately                                                      |
| Sub-period check  | Sharpe ratio in each half of the sample                                                                    |

| Metric        | Definition                                       |
| ------------- | ------------------------------------------------ |
| `AnnReturn`   | Annualised net return                            |
| `AnnVol`      | Annualised volatility                            |
| `Sharpe`      | Annualised excess-return Sharpe ratio            |
| `Sortino`     | Excess return over downside deviation            |
| `MaxDrawdown` | Largest peak-to-trough loss                      |
| `Calmar`      | Annualised return over absolute maximum drawdown |
| `CVaR5`       | Average of the worst 5% of monthly returns       |
| `AnnTurnover` | Average monthly turnover times 12                |

---

## Outputs

| Path               | Content                                                                 |
| ------------------ | ----------------------------------------------------------------------- |
| `data/processed/`  | Validated panel, regimes, monthly features, forecasts, weights, returns |
| `results/tables/`  | All metric tables as CSV                                                |
| `results/figures/` | Ten report figures                                                      |
| `reports/`         | `Kairos_Report.md` and `.docx`, `Kairos_Deck.pptx`                      |
| `notebooks/`       | Three notebooks that display the results tables and figures             |

The report, deck and notebooks read the results files, so no number is typed by hand.

| Notebook                    | Focus                                             |
| --------------------------- | ------------------------------------------------- |
| `01_data_and_regimes`       | Data quality, regime validation, regime timeline  |
| `02_forecasting`            | Information coefficients and SHAP importance      |
| `03_portfolio_and_backtest` | Performance, significance, regime and cost tables |

---

## Project layout

| Path         | Responsibility                                    |
| ------------ | ------------------------------------------------- |
| `config.py`  | Parameters, seeds and paths                       |
| `run.py`     | Command line entry point                          |
| `data/`      | Download, panel build and validation              |
| `features/`  | Causal feature engineering                        |
| `models/`    | Regimes, forecasting, LSTM, SHAP, Black-Litterman |
| `backtest/`  | Strategies, engine and bootstrap test             |
| `reporting/` | Figures, report, deck and notebooks               |
| `tests/`     | Unit and end-to-end tests                         |

---

## Key parameters

All parameters live in `config.py`, and the random seed is fixed at 690.

| Parameter              | Value | Meaning                                    |
| ---------------------- | ----- | ------------------------------------------ |
| `INITIAL_TRAIN_MONTHS` | 48    | Months used only for initial training      |
| `HMM_REFIT_EVERY`      | 252   | Trading days between HMM refits            |
| `STRESS_THRESHOLD`     | 0.5   | Probability cutoff for the stressed regime |
| `STACK_WINDOW`         | 24    | Months used to fit stacking weights        |
| `TAU`                  | 0.05  | Black-Litterman prior uncertainty          |
| `BASE_DELTA`           | 2.5   | Baseline risk aversion                     |
| `STRESSED_DELTA_MULT`  | 1.8   | Risk aversion multiplier in stress         |
| `STRESSED_OMEGA_MULT`  | 2.5   | View uncertainty multiplier in stress      |
| `MAX_WEIGHT`           | 0.35  | Per-asset position cap                     |
| `COV_WINDOW`           | 60    | Months in the covariance estimate          |
| `COST_BPS`             | 10    | Base transaction cost                      |

---

## Tests

Run the suite with `python -m pytest tests -q`. Tests use a factor-model fixture panel from `tests/fixtures.py`, never the real results.

| Test file                 | Coverage                                    |
| ------------------------- | ------------------------------------------- |
| `test_data.py`            | Panel building and validation rules         |
| `test_features.py`        | Causality of features                       |
| `test_regime.py`          | Causality and behaviour of regime detection |
| `test_lstm.py`            | Gradient correctness of the NumPy LSTM      |
| `test_forecasting.py`     | Walk-forward forecasting and stacking       |
| `test_black_litterman.py` | Black-Litterman algebra and optimiser       |
| `test_backtest.py`        | Timing, accounting, bootstrap calibration   |
| `test_pipeline.py`        | Full end-to-end run through report and deck |

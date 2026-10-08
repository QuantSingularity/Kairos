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

## Headline result

The central hypothesis is not supported on this sample. The forecasters have no out-of-sample skill (mean stacked IC -0.037, significantly negative for SPY and BTC-USD, with a negative R2 against the historical mean on every asset), and the Kairos portfolios do not beat static 60/40 or the market reference. Applying the same stress rule mechanically to 60/40 matches its Sharpe ratio and cuts its drawdown, so the useful part of Kairos is the regime-driven de-risking, not the forecasts. None of 42 Black-Litterman parameter settings reaches the 60/40 Sharpe ratio. The three Kairos variants were tried on one sample, so the best of them is exploratory.

| Strategy             | Sharpe | Ann. return | Ann. vol | Max drawdown |
| -------------------- | ------ | ----------- | -------- | ------------ |
| `BL_ML_Regime`       | 0.78   | 7.7%        | 8.2%     | -16.3%       |
| `BL_ML_Regime_IC`    | 0.86   | 7.5%        | 7.1%     | -13.3%       |
| `BL_HistMean_Regime` | 0.89   | 8.0%        | 7.4%     | -15.3%       |
| `Static_60_40`       | 0.98   | 10.4%       | 9.1%     | -20.5%       |
| `Regime_60_40`       | 0.98   | 8.2%        | 6.9%     | -13.1%       |
| `Market_Reference`   | 0.93   | 10.4%       | 9.7%     | -21.1%       |

Figures come from the committed run on data through September 2026 and regenerate with `python run.py all`. The full discussion is in `reports/Kairos_Report.md`.

---

## Quick start

```
pip install -r requirements.txt
python run.py all
```

The first run downloads prices from Yahoo Finance and caches them in `data/raw/`. Later runs reuse the cache. You can also place your own `Date,Close` CSV files there, named like `SPY.csv`, `BTC_USD.csv`, `VIX.csv` and `IRX.csv`. If Pandoc is installed, the report is also converted to Word. The notebooks are executed when `nbclient` and `ipykernel` are installed, so they ship with saved outputs.

| Command                 | Purpose                                                    |
| ----------------------- | ---------------------------------------------------------- |
| `python run.py all`     | Run every stage in order                                   |
| `python run.py <stage>` | Run a single stage                                         |
| `--refresh`             | Download prices again instead of using the cache           |
| `--assets SPY TLT`      | Restrict forecasting to the listed tickers                 |
| `--jobs N`              | Number of parallel workers for forecasting and diagnostics |

---

## Pipeline

| #   | Stage         | What it does                                                      | Main output                                                    |
| --- | ------------- | ----------------------------------------------------------------- | -------------------------------------------------------------- |
| 1   | `data`        | Downloads, aligns and validates the price panel                   | `data/processed/panel_daily.csv`                               |
| 2   | `regimes`     | Fits the walk-forward HMM and the volatility rule                 | `data/processed/regimes_daily.csv`                             |
| 3   | `features`    | Builds causal monthly features                                    | `data/processed/features_monthly.csv`                          |
| 4   | `forecast`    | Runs walk-forward forecasts and skill metrics for every asset     | `data/processed/forecasts_<asset>.csv`                         |
| 5   | `diagnostics` | Re-fits tree and linear models under alternative designs          | `results/tables/ic_diagnostics*.csv`, `regime_return_link.csv` |
| 6   | `shap`        | Explains the forecasts by regime                                  | `results/tables/shap_<asset>.csv`                              |
| 7   | `portfolio`   | Builds portfolios, backtests, tests significance, deflates Sharpe | `results/tables/*.csv`                                         |
| 8   | `sensitivity` | Changes one Black-Litterman parameter at a time                   | `results/tables/sensitivity.csv`                               |
| 9   | `figures`     | Draws the fourteen report figures                                 | `results/figures/*.png`                                        |
| 10  | `report`      | Writes the report in Markdown and Word                            | `reports/Kairos_Report.md`                                     |
| 11  | `deck`        | Builds the slide deck                                             | `reports/Kairos_Deck.pptx`                                     |
| 12  | `notebooks`   | Writes and executes four results notebooks                        | `notebooks/*.ipynb`                                            |

The diagnostics stage takes about ten minutes on a single core. The forecast stage is the slowest stage overall.

---

## Data

Daily adjusted closes come from Yahoo Finance between 2003 and September 2026. Nothing is simulated.

| Ticker  | Role                                       | Prior weight |
| ------- | ------------------------------------------ | ------------ |
| SPY     | US large-cap equities                      | 42%          |
| EFA     | International developed equities           | 12%          |
| IEF     | Intermediate US Treasuries                 | 15%          |
| TLT     | Long US Treasuries                         | 10%          |
| GLD     | Gold                                       | 8%           |
| DBC     | Broad commodities                          | 5%           |
| HYG     | High-yield credit                          | 5%           |
| BTC-USD | Bitcoin                                    | 2%           |
| ETH-USD | Ethereum                                   | 1%           |
| ^VIX    | Volatility input for regimes and features  | n/a          |
| ^IRX    | 13-week T-bill, used as the risk-free rate | n/a          |

The prior weights are a hand-picked reference allocation, not an observed market-capitalisation series. The `prior=equal` sensitivity run shows how much the results depend on them.

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
| Skill metrics | Information coefficient, hit rate against the always-up baseline, out-of-sample R2 against an expanding historical mean, and forecast dispersion  |

### Forecast diagnostics

| Item        | Setting                                                                                                                           |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------- |
| Models      | XGBoost, Random Forest and Ridge, pooled by equal weight                                                                          |
| Designs     | As specified, no regime features, no stress features (regime and VIX), 72-month initial window, no regime features with 72 months |
| Comparison  | Dates common to all designs, so differences are not driven by sample length                                                       |
| Regime link | Rank correlation of the stress probability with next-month return, inside and after the initial window                            |
| Status      | Exploratory. Uses the same sample as the main test, so no design is adopted into the main pipeline                                |

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

| Variant              | Description                                                                           |
| -------------------- | ------------------------------------------------------------------------------------- |
| `BL_ML_Regime`       | Black-Litterman with regime scaling                                                   |
| `BL_ML_Regime_IC`    | Regime scaling plus views scaled by each asset's trailing IC                          |
| `BL_ML_NoRegime`     | Same engine without regime scaling, used as the ablation                              |
| `BL_HistMean_Regime` | Regime scaling with expanding historical-mean views instead of machine learning views |

### Benchmarks

| Benchmark          | Description                                                                                                          |
| ------------------ | -------------------------------------------------------------------------------------------------------------------- |
| `Market_Reference` | Reference market weights                                                                                             |
| `Equal_Weight`     | Equal weight across the eligible universe                                                                            |
| `Static_60_40`     | 60% SPY and 40% IEF                                                                                                  |
| `Inverse_Vol`      | Weights proportional to inverse 36-month volatility                                                                  |
| `MeanVariance_LW`  | Mean-variance optimisation with Ledoit-Wolf covariance                                                               |
| `Regime_60_40`     | Static 60/40 with risky weights divided by the stress risk-aversion multiplier in stressed months, remainder in cash |
| `Regime_Market`    | The same mechanical stress rule applied to the market reference                                                      |

All benchmarks use the same eligible universe as the Kairos portfolios. The two regime overlays use no forecasts, so they isolate what the machine learning and Black-Litterman layers add beyond a defensive exposure cut.

---

## Evaluation

| Component         | Detail                                                                                                              |
| ----------------- | ------------------------------------------------------------------------------------------------------------------- |
| Rebalancing       | Monthly, with weight drift between rebalances                                                                       |
| Transaction costs | 10 bps on traded notional, with a 0, 10, 25 and 50 bps grid                                                         |
| Cash return       | T-bill rate                                                                                                         |
| Sharpe ratio      | Measured on excess returns over the T-bill rate                                                                     |
| Significance      | Paired stationary bootstrap of the Sharpe difference, centred on the null, 4,000 draws with 6-month blocks          |
| Regime split      | Calm and stressed decision months reported separately                                                               |
| Sub-period check  | Sharpe ratio in each half of the sample                                                                             |
| Deflated Sharpe   | Probability the Sharpe ratio is positive after allowing for 3 and 10 trials (tests against zero, not against 60/40) |
| Sensitivity       | One-at-a-time grid over `tau`, `base_delta`, the stress multipliers, the position cap and the prior                 |

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
| `results/figures/` | Fourteen report figures                                                 |
| `reports/`         | `Kairos_Report.md` and `.docx`, `Kairos_Deck.pptx`                      |
| `notebooks/`       | Four executed notebooks that display the results tables and figures     |

The report, deck and notebooks read the results files, so no number is typed by hand.

| Notebook                        | Focus                                                               |
| ------------------------------- | ------------------------------------------------------------------- |
| `01_data_and_regimes`           | Data quality, regime validation, regime timeline                    |
| `02_forecasting`                | Information coefficients and SHAP importance                        |
| `03_portfolio_and_backtest`     | Performance, significance, regime and cost tables                   |
| `04_diagnostics_and_robustness` | Regime link, IC diagnostics, deflated Sharpe, overlays, sensitivity |

---

## Project layout

| Path         | Responsibility                                                 |
| ------------ | -------------------------------------------------------------- |
| `config.py`  | Parameters, seeds and paths                                    |
| `run.py`     | Command line entry point                                       |
| `data/`      | Download, panel build and validation                           |
| `features/`  | Causal feature engineering                                     |
| `models/`    | Regimes, forecasting, diagnostics, LSTM, SHAP, Black-Litterman |
| `backtest/`  | Strategies, engine, bootstrap test, sensitivity                |
| `reporting/` | Figures, report, deck and notebooks                            |
| `tests/`     | Unit and end-to-end tests                                      |

---

## Key parameters

All parameters live in `config.py`, and the random seed is fixed at 690.

| Parameter              | Value    | Meaning                                         |
| ---------------------- | -------- | ----------------------------------------------- |
| `INITIAL_TRAIN_MONTHS` | 48       | Months used only for initial training           |
| `HMM_REFIT_EVERY`      | 252      | Trading days between HMM refits                 |
| `STRESS_THRESHOLD`     | 0.5      | Probability cutoff for the stressed regime      |
| `STACK_WINDOW`         | 24       | Months used to fit stacking weights             |
| `TAU`                  | 0.05     | Black-Litterman prior uncertainty               |
| `BASE_DELTA`           | 2.5      | Baseline risk aversion                          |
| `STRESSED_DELTA_MULT`  | 1.8      | Risk aversion multiplier in stress              |
| `STRESSED_OMEGA_MULT`  | 2.5      | View uncertainty multiplier in stress           |
| `MAX_WEIGHT`           | 0.35     | Per-asset position cap                          |
| `COV_WINDOW`           | 60       | Months in the covariance estimate               |
| `COST_BPS`             | 10       | Base transaction cost                           |
| `DIAG_INITIAL_TRAIN`   | 72       | Initial window in the longer-window diagnostic  |
| `DSR_TRIALS`           | 3, 10    | Trial counts used for the deflated Sharpe ratio |
| `SENSITIVITY_GRID`     | see file | Parameter values in the sensitivity runs        |

Dependencies are pinned in `requirements.txt`.

---

## Tests

Run the suite with `python -m pytest tests -q`. Tests use a factor-model fixture panel from `tests/fixtures.py`, never the real results.

| Test file                 | Coverage                                                                   |
| ------------------------- | -------------------------------------------------------------------------- |
| `test_data.py`            | Panel building and validation rules                                        |
| `test_features.py`        | Causality of features                                                      |
| `test_regime.py`          | Causality and behaviour of regime detection                                |
| `test_lstm.py`            | Gradient correctness of the NumPy LSTM                                     |
| `test_forecasting.py`     | Walk-forward forecasting and stacking                                      |
| `test_new_features.py`    | Deflated Sharpe, overlay, parameters, R2 and hit-rate metrics, diagnostics |
| `test_hygiene.py`         | No long dashes, emojis or patch-log comments in text files                 |
| `test_black_litterman.py` | Black-Litterman algebra and optimiser                                      |
| `test_backtest.py`        | Timing, accounting, bootstrap calibration                                  |
| `test_pipeline.py`        | Full end-to-end run through report and deck                                |

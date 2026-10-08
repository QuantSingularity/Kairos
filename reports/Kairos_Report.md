# Kairos: Regime-Aware Portfolio Intelligence

Track: Machine Learning (Deep Learning) Investment Strategies, Topic 2: Portfolio Allocation.

## 1. Research Question

Does a machine learning and deep learning driven approach to portfolio allocation, augmented with market regime awareness, produce statistically and economically meaningful improvements in risk-adjusted returns over classical allocation methods, and under what market conditions does this advantage hold or break down?

## 2. Summary of Findings

The research hypothesis is not supported. The highest Sharpe ratio among the Kairos portfolios and the classical benchmarks belongs to Static_60_40. The Sharpe difference of BL_ML_Regime to the best classical benchmark (Static_60_40) is -0.20 (95% interval -0.53 to 0.07, p = 0.197), not statistically significant; BL_ML_Regime is significantly better than 1 and significantly worse than 0 of the 5 classical benchmarks (5% level). For BL_ML_Regime_IC the difference is -0.12 (95% interval -0.34 to 0.05, p = 0.233), not statistically significant.

Regime ablation: the Sharpe difference between BL_ML_Regime and BL_ML_NoRegime is +0.10 (95% interval -0.03 to 0.21, p = 0.119), not statistically significant. In stressed months the regime-aware portfolio's worst-5% average monthly return was -4.67% against -6.83% without regime scaling.

Forecasting: the stacked ensemble has a mean out-of-sample information coefficient of -0.037. 2 of 9 assets show an IC that is significantly negative at the 5% level (SPY, BTC-USD) and 0 show one that is significantly positive. The stacked ensemble beats the mean of the single models on 4 of 9 assets.

Overlay ablation: the same stress rule applied to static 60/40 gives a Sharpe ratio of 0.98 against 0.98, with a maximum drawdown of -13.1% against -20.5%.

Diagnosis: the stress-to-return relationship changes sign between the initial training window and later years for 6 of 9 assets, and the R2 against the historical mean is negative under every re-fitted design.

Robustness: 0 of 42 Black-Litterman parameter settings exceed the Sharpe ratio of static 60/40.

Regimes: the walk-forward HMM flags 84% of days inside the documented stress windows and 8% of days inside the calm control windows, against 66% and 7% for the volatility rule.

Sharpe ranking: Static_60_40 (0.98), Regime_60_40 (0.98), Market_Reference (0.93), Regime_Market (0.91), BL_HistMean_Regime (0.89), BL_ML_Regime_IC (0.86), BL_ML_Regime (0.78), Inverse_Vol (0.75), Equal_Weight (0.74), BL_ML_NoRegime (0.68), MeanVariance_LW (0.40).

## 3. Data

All prices are daily adjusted closes downloaded from Yahoo Finance through the yfinance library, with the CBOE VIX (^VIX) and the 13-week Treasury bill yield (^IRX) as the volatility and risk-free inputs. No series is simulated, interpolated or reconstructed. The panel is validated before use: history length, gaps, extreme daily moves, and sanity floors on cross-asset correlations (for example SPY and EFA) that would expose substitute or corrupted data.

| series  | first_date | observations |
| ------- | ---------- | ------------ |
| SPY     | 2003-01-02 | 6194         |
| EFA     | 2003-01-02 | 6194         |
| TLT     | 2003-01-02 | 6194         |
| IEF     | 2003-01-02 | 6194         |
| HYG     | 2007-04-11 | 5080         |
| GLD     | 2004-11-18 | 5704         |
| DBC     | 2006-02-06 | 5387         |
| BTC-USD | 2014-09-17 | 3140         |
| ETH-USD | 2017-11-09 | 2319         |
| VIX     | 2003-01-02 | 6194         |
| RF      | 2003-01-02 | 6194         |

The investable universe expands over time. An asset enters the portfolio only after it has a valid walk-forward forecast and at least 24 months of return history, so crypto assets enter when they have a usable track record.

## 4. Method

**Features.** About 20 causal technical, volatility, momentum, drawdown and market-beta features per asset, computed from daily prices, plus VIX level, change and z-score. Raw price levels are never used as features. Features are sampled at month end.

**Regimes.** A two-state Gaussian HMM on SPY return, 21-day realised volatility and VIX. It is refit once a year on past data only, and daily states come from the forward filter, so the label on any day uses no later information. A volatility-percentile rule is the comparison method.

**Forecasting.** One-month-ahead log return per asset with walk-forward XGBoost, Random Forest, Ridge and a NumPy LSTM, plus an equal-weight ensemble and a stacked ensemble whose non-negative weights are fit on the previous 24 months. The first 48 months are used only for initial training. Skill is judged by the information coefficient, the hit rate against the always-up baseline, and the out-of-sample R2 against an expanding historical-mean forecast.

**Forecast diagnostics.** The tree and linear models are re-fitted without the regime features, without all market-wide stress inputs (regime and VIX), with a 72-month initial window, and with the first and last changes combined, and compared on the dates common to all designs. The link between the stress probability and next-month returns is measured separately inside and after the initial window.

**Portfolio construction.** Black-Litterman with a Ledoit-Wolf covariance, a reference market-weight prior, and the stacked forecasts as absolute views. View uncertainty follows the trailing forecast error variance. In the stressed regime, view uncertainty is multiplied and risk aversion is raised. The optimiser allows cash, so higher risk aversion lowers exposure. BL_ML_Regime_IC additionally scales each view by the asset's trailing IC. BL_ML_NoRegime removes the regime scaling and serves as the ablation. BL_HistMean_Regime replaces the machine learning views with the expanding historical mean.

**Benchmarks.** Market reference weights, equal weight, 60/40 (SPY and IEF), inverse volatility, and mean-variance with Ledoit-Wolf covariance. All benchmarks use the same eligible universe as the Black-Litterman portfolios. Regime_60_40 and Regime_Market apply the stress rule mechanically to 60/40 and to the market reference, with no forecasts.

**Backtest.** Monthly rebalancing, 10 basis points on traded notional, weight drift, cash earning the T-bill rate, and Sharpe ratios measured on excess returns over that rate. Significance uses a paired stationary bootstrap of the Sharpe difference, centred on the null. A deflated Sharpe ratio allows for the Kairos variants tried. Sensitivity runs change one Black-Litterman parameter at a time.

## 5. Results

### 5.1 Regime detection

| Period                       | Kind   | Days_hmm | FlaggedStressed_hmm | Rate_hmm | Days_vol | FlaggedStressed_vol | Rate_vol |
| ---------------------------- | ------ | -------- | ------------------- | -------- | -------- | ------------------- | -------- |
| 2008 Global Financial Crisis | stress | 152      | 152                 | 100.0%   | 152      | 70                  | 46.1%    |
| 2011 European Debt Crisis    | stress | 66       | 54                  | 81.8%    | 66       | 54                  | 81.8%    |
| 2015-2016 China Growth Scare | stress | 131      | 63                  | 48.1%    | 131      | 82                  | 62.6%    |
| 2018 Q4 Selloff              | stress | 61       | 53                  | 86.9%    | 61       | 35                  | 57.4%    |
| 2020 COVID Crash             | stress | 35       | 32                  | 91.4%    | 35       | 32                  | 91.4%    |
| 2022 Rate-Hiking Selloff     | stress | 205      | 191                 | 93.2%    | 205      | 112                 | 54.6%    |
| 2013-2014 Calm               | calm   | 521      | 16                  | 3.1%     | 521      | 74                  | 14.2%    |
| 2017 Low Volatility          | calm   | 259      | 0                   | 0.0%     | 259      | 13                  | 5.0%     |
| 2021 Recovery                | calm   | 197      | 43                  | 21.8%    | 197      | 3                   | 1.5%     |

![Share of days flagged stressed](../results/figures/03_regime_validation.png)

_Share of days flagged stressed_

![Walk-forward regime timeline](../results/figures/02_regime_timeline.png)

_Walk-forward regime timeline_

### 5.2 Forecasting skill

Out-of-sample information coefficient by asset and model:

| Asset   | IC_xgb | IC_rf  | IC_ridge | IC_lstm | IC_ensemble_equal | IC_ensemble_stacked | p_value_stacked |
| ------- | ------ | ------ | -------- | ------- | ----------------- | ------------------- | --------------- |
| SPY     | -0.047 | -0.106 | -0.128   | 0.007   | -0.042            | -0.160              | 0.020           |
| EFA     | -0.151 | -0.163 | -0.215   | -0.022  | -0.118            | -0.112              | 0.106           |
| TLT     | -0.036 | 0.039  | -0.042   | 0.035   | 0.028             | -0.046              | 0.506           |
| IEF     | 0.113  | 0.063  | -0.049   | 0.035   | 0.065             | -0.021              | 0.761           |
| HYG     | 0.081  | 0.106  | -0.051   | -0.041  | 0.021             | 0.030               | 0.691           |
| GLD     | 0.041  | 0.029  | -0.104   | -0.106  | -0.059            | 0.029               | 0.678           |
| DBC     | 0.047  | 0.038  | 0.094    | -0.020  | 0.023             | 0.037               | 0.616           |
| BTC-USD | -0.110 | -0.122 | -0.093   | -0.153  | -0.218            | -0.302              | 0.006           |
| ETH-USD | -0.029 | 0.018  | -0.145   | 0.252   | 0.141             | 0.213               | 0.156           |

![Out-of-sample IC by asset and model](../results/figures/04_model_ic.png)

_Out-of-sample IC by asset and model_

Hit rate against the always-up baseline and R2 against the historical mean (HitEdge is the hit rate minus the share of up months):

| Asset   | BaseRate | UpShare_stacked | Hit_ensemble_stacked | HitEdge_ensemble_stacked | IC_ensemble_stacked | R2OOS_ensemble_stacked | R2OOS_xgb | R2OOS_rf | R2OOS_ridge | R2OOS_lstm |
| ------- | -------- | --------------- | -------------------- | ------------------------ | ------------------- | ---------------------- | --------- | -------- | ----------- | ---------- |
| SPY     | 0.687    | 0.834           | 0.597                | -0.090                   | -0.160              | -0.201                 | -0.262    | -0.175   | -0.052      | -3.751     |
| EFA     | 0.616    | 0.716           | 0.498                | -0.118                   | -0.112              | -0.155                 | -0.342    | -0.253   | -0.029      | -3.442     |
| TLT     | 0.507    | 0.564           | 0.441                | -0.066                   | -0.046              | -0.391                 | -0.188    | -0.078   | -0.070      | -5.792     |
| IEF     | 0.521    | 0.706           | 0.436                | -0.085                   | -0.021              | -0.213                 | -0.060    | -0.045   | -0.033      | -6.947     |
| HYG     | 0.665    | 0.861           | 0.607                | -0.058                   | 0.030               | -0.108                 | -0.095    | -0.101   | -0.304      | -6.711     |
| GLD     | 0.512    | 0.701           | 0.428                | -0.085                   | 0.029               | -0.083                 | -0.148    | -0.068   | -0.003      | -2.778     |
| DBC     | 0.524    | 0.444           | 0.428                | -0.096                   | 0.037               | -0.169                 | -0.179    | -0.076   | -0.017      | -2.559     |
| BTC-USD | 0.554    | 0.735           | 0.373                | -0.181                   | -0.302              | -0.169                 | -0.294    | -0.162   | -0.013      | -1.143     |
| ETH-USD | 0.457    | 0.717           | 0.522                | 0.065                    | 0.213               | -0.041                 | -0.329    | -0.087   | 0.000       | -1.190     |

![Forecast skill of the stacked ensemble](../results/figures/11_forecast_skill.png)

_Forecast skill of the stacked ensemble_

![Regime-conditional SHAP importance (in-sample, for explanation only)](../results/figures/05_shap_regime.png)

_Regime-conditional SHAP importance (in-sample, for explanation only)_

### 5.3 Diagnosing the information coefficient

Rank correlation between the filtered stress probability and the next-month return, inside and after the initial training window:

| Asset   | Window         | n   | CorrStressNextRet | MeanRetStressed | MeanRetCalm | StressedMonths |
| ------- | -------------- | --- | ----------------- | --------------- | ----------- | -------------- |
| BTC-USD | initial_window | 48  | -0.151            | -0.033          | 0.106       | 11             |
| BTC-USD | after          | 83  | 0.186             | 0.060           | 0.006       | 32             |
| DBC     | initial_window | 48  | -0.143            | 0.003           | 0.012       | 39             |
| DBC     | after          | 187 | 0.173             | 0.004           | 0.000       | 51             |
| EFA     | initial_window | 48  | -0.324            | -0.024          | 0.016       | 24             |
| EFA     | after          | 211 | 0.198             | 0.015           | 0.003       | 69             |
| ETH-USD | initial_window | 48  | 0.043             | 0.045           | 0.040       | 28             |
| ETH-USD | after          | 46  | 0.318             | 0.092           | -0.010      | 9              |
| GLD     | initial_window | 48  | -0.030            | 0.017           | 0.021       | 34             |
| GLD     | after          | 201 | 0.120             | 0.006           | 0.006       | 59             |
| HYG     | initial_window | 48  | 0.025             | 0.008           | 0.002       | 33             |
| HYG     | after          | 173 | 0.079             | 0.004           | 0.004       | 46             |
| IEF     | initial_window | 48  | 0.188             | 0.011           | 0.000       | 24             |
| IEF     | after          | 211 | 0.010             | 0.001           | 0.002       | 69             |
| SPY     | initial_window | 48  | -0.228            | -0.020          | 0.008       | 24             |
| SPY     | after          | 211 | 0.206             | 0.018           | 0.009       | 69             |
| TLT     | initial_window | 48  | 0.070             | 0.013           | -0.001      | 24             |
| TLT     | after          | 211 | -0.055            | -0.001          | 0.003       | 69             |

Tree and linear models re-fitted under alternative designs, compared on common dates:

| Variant                 | MeanIC | MeanIC_equities | MeanR2OOS | AssetsPositiveIC | Assets |
| ----------------------- | ------ | --------------- | --------- | ---------------- | ------ |
| as_specified            | -0.011 | -0.064          | -0.065    | 3                | 9      |
| no_regime_features      | -0.034 | -0.097          | -0.070    | 3                | 9      |
| no_stress_features      | 0.018  | -0.048          | -0.050    | 6                | 9      |
| longer_initial_window   | -0.008 | -0.085          | -0.064    | 3                | 9      |
| no_regime_longer_window | -0.034 | -0.097          | -0.070    | 3                | 9      |

![Mean IC by forecasting design](../results/figures/12_ic_diagnostics.png)

_Mean IC by forecasting design_

### 5.4 Portfolio performance

| index              | AnnReturn | AnnVol | Sharpe | Sortino | MaxDrawdown | Calmar | CVaR5  | AnnTurnover |
| ------------------ | --------- | ------ | ------ | ------- | ----------- | ------ | ------ | ----------- |
| BL_ML_Regime       | 7.70%     | 8.23%  | 0.780  | 1.218   | -16.28%     | 0.473  | -5.03% | 4.37        |
| BL_ML_Regime_IC    | 7.47%     | 7.09%  | 0.863  | 1.374   | -13.30%     | 0.562  | -3.87% | 1.92        |
| BL_ML_NoRegime     | 7.56%     | 9.34%  | 0.684  | 1.028   | -22.83%     | 0.331  | -6.26% | 5.03        |
| BL_HistMean_Regime | 7.96%     | 7.38%  | 0.894  | 1.458   | -15.31%     | 0.520  | -3.89% | 1.34        |
| Market_Reference   | 10.39%    | 9.74%  | 0.927  | 1.519   | -21.06%     | 0.494  | -5.56% | 0.35        |
| Equal_Weight       | 9.49%     | 11.25% | 0.744  | 1.300   | -23.59%     | 0.402  | -5.89% | 0.54        |
| Static_60_40       | 10.37%    | 9.12%  | 0.980  | 1.596   | -20.52%     | 0.505  | -5.19% | 0.27        |
| Inverse_Vol        | 6.64%     | 7.13%  | 0.749  | 1.185   | -17.73%     | 0.374  | -4.19% | 0.46        |
| MeanVariance_LW    | 6.00%     | 13.28% | 0.402  | 0.617   | -36.97%     | 0.162  | -8.11% | 2.52        |
| Regime_60_40       | 8.23%     | 6.95%  | 0.980  | 1.552   | -13.15%     | 0.626  | -3.86% | 1.19        |
| Regime_Market      | 8.12%     | 7.42%  | 0.910  | 1.456   | -13.62%     | 0.597  | -3.99% | 1.25        |

![Cumulative net return](../results/figures/07_cumulative_returns.png)

_Cumulative net return_

![Drawdowns](../results/figures/08_drawdowns.png)

_Drawdowns_

![Portfolio weights](../results/figures/06_weights.png)

_Portfolio weights_

### 5.5 Statistical significance

| strategy        | benchmark          | diff   | ci_low | ci_high | p_value |
| --------------- | ------------------ | ------ | ------ | ------- | ------- |
| BL_ML_Regime    | BL_ML_Regime_IC    | -0.083 | -0.286 | 0.114   | 0.412   |
| BL_ML_Regime    | BL_ML_NoRegime     | 0.096  | -0.032 | 0.206   | 0.119   |
| BL_ML_Regime    | BL_HistMean_Regime | -0.114 | -0.351 | 0.101   | 0.333   |
| BL_ML_Regime    | Market_Reference   | -0.147 | -0.406 | 0.076   | 0.230   |
| BL_ML_Regime    | Equal_Weight       | 0.036  | -0.276 | 0.347   | 0.818   |
| BL_ML_Regime    | Static_60_40       | -0.201 | -0.528 | 0.074   | 0.197   |
| BL_ML_Regime    | Inverse_Vol        | 0.031  | -0.288 | 0.295   | 0.834   |
| BL_ML_Regime    | MeanVariance_LW    | 0.378  | 0.016  | 0.752   | 0.045   |
| BL_ML_Regime    | Regime_60_40       | -0.200 | -0.539 | 0.099   | 0.227   |
| BL_ML_Regime    | Regime_Market      | -0.131 | -0.371 | 0.108   | 0.286   |
| BL_ML_Regime_IC | BL_ML_Regime       | 0.083  | -0.114 | 0.286   | 0.412   |
| BL_ML_Regime_IC | BL_ML_NoRegime     | 0.179  | -0.068 | 0.408   | 0.144   |
| BL_ML_Regime_IC | BL_HistMean_Regime | -0.031 | -0.149 | 0.092   | 0.617   |
| BL_ML_Regime_IC | Market_Reference   | -0.064 | -0.197 | 0.054   | 0.335   |
| BL_ML_Regime_IC | Equal_Weight       | 0.119  | -0.129 | 0.407   | 0.367   |
| BL_ML_Regime_IC | Static_60_40       | -0.117 | -0.335 | 0.053   | 0.233   |
| BL_ML_Regime_IC | Inverse_Vol        | 0.114  | -0.147 | 0.343   | 0.355   |
| BL_ML_Regime_IC | MeanVariance_LW    | 0.461  | 0.103  | 0.820   | 0.012   |
| BL_ML_Regime_IC | Regime_60_40       | -0.117 | -0.315 | 0.049   | 0.203   |
| BL_ML_Regime_IC | Regime_Market      | -0.047 | -0.115 | 0.015   | 0.144   |
| Regime_60_40    | Static_60_40       | -0.001 | -0.137 | 0.125   | 0.994   |
| Regime_Market   | Market_Reference   | -0.017 | -0.143 | 0.098   | 0.795   |
| Regime_60_40    | BL_ML_Regime       | 0.200  | -0.099 | 0.539   | 0.227   |
| Regime_Market   | BL_ML_Regime       | 0.131  | -0.108 | 0.371   | 0.286   |

![Sharpe difference with 95% bootstrap interval](../results/figures/09_sharpe_significance.png)

_Sharpe difference with 95% bootstrap interval_

Deflated Sharpe ratio of the Kairos variants (DSR is the probability that the true Sharpe ratio is positive after allowing for the number of trials):

| strategy        | Sharpe | BenchmarkSharpe | DSR   | Trials |
| --------------- | ------ | --------------- | ----- | ------ |
| BL_ML_Regime    | 0.780  | 0.077           | 0.997 | 3      |
| BL_ML_Regime    | 0.780  | 0.141           | 0.994 | 10     |
| BL_ML_Regime_IC | 0.863  | 0.077           | 0.999 | 3      |
| BL_ML_Regime_IC | 0.863  | 0.141           | 0.998 | 10     |
| BL_ML_NoRegime  | 0.684  | 0.077           | 0.991 | 3      |
| BL_ML_NoRegime  | 0.684  | 0.141           | 0.983 | 10     |

### 5.6 Overlay ablation

| index              | AnnReturn | AnnVol | Sharpe | Sortino | MaxDrawdown | Calmar | CVaR5  | AnnTurnover |
| ------------------ | --------- | ------ | ------ | ------- | ----------- | ------ | ------ | ----------- |
| BL_ML_Regime       | 7.70%     | 8.23%  | 0.780  | 1.218   | -16.28%     | 0.473  | -5.03% | 4.37        |
| BL_ML_Regime_IC    | 7.47%     | 7.09%  | 0.863  | 1.374   | -13.30%     | 0.562  | -3.87% | 1.92        |
| BL_HistMean_Regime | 7.96%     | 7.38%  | 0.894  | 1.458   | -15.31%     | 0.520  | -3.89% | 1.34        |
| Regime_60_40       | 8.23%     | 6.95%  | 0.980  | 1.552   | -13.15%     | 0.626  | -3.86% | 1.19        |
| Static_60_40       | 10.37%    | 9.12%  | 0.980  | 1.596   | -20.52%     | 0.505  | -5.19% | 0.27        |
| Regime_Market      | 8.12%     | 7.42%  | 0.910  | 1.456   | -13.62%     | 0.597  | -3.99% | 1.25        |
| Market_Reference   | 10.39%    | 9.74%  | 0.927  | 1.519   | -21.06%     | 0.494  | -5.56% | 0.35        |

![Forecast-driven portfolios against mechanical overlays](../results/figures/14_overlay_ablation.png)

_Forecast-driven portfolios against mechanical overlays_

### 5.7 Performance by regime

| index                         | AnnReturn | Sharpe | MaxDrawdown | CVaR5  | Months |
| ----------------------------- | --------- | ------ | ----------- | ------ | ------ |
| BL_ML_Regime / All            | 7.70%     | 0.780  | -16.28%     | -5.03% | 211    |
| BL_ML_Regime / Calm           | 7.40%     | 0.712  | -10.81%     | -4.95% | 142    |
| BL_ML_Regime / Stressed       | 8.31%     | 0.925  | -11.11%     | -4.67% | 69     |
| BL_ML_Regime_IC / All         | 7.47%     | 0.863  | -13.30%     | -3.87% | 211    |
| BL_ML_Regime_IC / Calm        | 6.83%     | 0.748  | -10.30%     | -3.75% | 142    |
| BL_ML_Regime_IC / Stressed    | 8.81%     | 1.101  | -9.73%      | -3.91% | 69     |
| BL_ML_NoRegime / All          | 7.56%     | 0.684  | -22.83%     | -6.26% | 211    |
| BL_ML_NoRegime / Calm         | 7.42%     | 0.714  | -10.78%     | -4.95% | 142    |
| BL_ML_NoRegime / Stressed     | 7.86%     | 0.647  | -18.02%     | -6.83% | 69     |
| BL_HistMean_Regime / All      | 7.96%     | 0.894  | -15.31%     | -3.89% | 211    |
| BL_HistMean_Regime / Calm     | 7.32%     | 0.786  | -10.91%     | -3.65% | 142    |
| BL_HistMean_Regime / Stressed | 9.29%     | 1.115  | -10.92%     | -4.06% | 69     |
| Market_Reference / All        | 10.39%    | 0.927  | -21.06%     | -5.56% | 211    |
| Market_Reference / Calm       | 7.60%     | 0.816  | -10.56%     | -3.88% | 142    |
| Market_Reference / Stressed   | 16.38%    | 1.147  | -18.30%     | -7.35% | 69     |
| Equal_Weight / All            | 9.49%     | 0.744  | -23.59%     | -5.89% | 211    |
| Equal_Weight / Calm           | 5.70%     | 0.471  | -13.24%     | -4.20% | 142    |
| Equal_Weight / Stressed       | 17.72%    | 1.169  | -21.16%     | -7.46% | 69     |
| Static_60_40 / All            | 10.37%    | 0.980  | -20.52%     | -5.19% | 211    |
| Static_60_40 / Calm           | 8.04%     | 0.922  | -9.97%      | -3.84% | 142    |
| Static_60_40 / Stressed       | 15.34%    | 1.141  | -17.20%     | -6.81% | 69     |
| Inverse_Vol / All             | 6.64%     | 0.749  | -17.73%     | -4.19% | 211    |
| Inverse_Vol / Calm            | 5.53%     | 0.675  | -8.59%      | -2.92% | 142    |
| Inverse_Vol / Stressed        | 8.95%     | 0.885  | -16.06%     | -5.29% | 69     |
| MeanVariance_LW / All         | 6.00%     | 0.402  | -36.97%     | -8.11% | 211    |
| MeanVariance_LW / Calm        | 3.88%     | 0.256  | -21.24%     | -7.21% | 142    |
| MeanVariance_LW / Stressed    | 10.48%    | 0.628  | -31.61%     | -8.96% | 69     |
| Regime_60_40 / All            | 8.23%     | 0.980  | -13.15%     | -3.86% | 211    |
| Regime_60_40 / Calm           | 7.96%     | 0.912  | -10.08%     | -3.85% | 142    |
| Regime_60_40 / Stressed       | 8.80%     | 1.117  | -9.44%      | -3.75% | 69     |
| Regime_Market / All           | 8.12%     | 0.910  | -13.62%     | -3.99% | 211    |
| Regime_Market / Calm          | 7.52%     | 0.806  | -10.67%     | -3.88% | 142    |
| Regime_Market / Stressed      | 9.38%     | 1.124  | -10.09%     | -4.05% | 69     |

![Annualised return by regime](../results/figures/10_regime_performance.png)

_Annualised return by regime_

### 5.8 Robustness

Sharpe ratio by transaction cost assumption:

| index              | 0bps  | 10bps | 25bps | 50bps |
| ------------------ | ----- | ----- | ----- | ----- |
| BL_ML_Regime       | 0.833 | 0.780 | 0.700 | 0.567 |
| BL_ML_Regime_IC    | 0.891 | 0.863 | 0.822 | 0.753 |
| BL_ML_NoRegime     | 0.738 | 0.684 | 0.603 | 0.468 |
| BL_HistMean_Regime | 0.913 | 0.894 | 0.866 | 0.818 |
| Market_Reference   | 0.931 | 0.927 | 0.921 | 0.911 |
| Equal_Weight       | 0.749 | 0.744 | 0.736 | 0.724 |
| Static_60_40       | 0.984 | 0.980 | 0.975 | 0.967 |
| Inverse_Vol        | 0.756 | 0.749 | 0.738 | 0.721 |
| MeanVariance_LW    | 0.421 | 0.402 | 0.374 | 0.326 |
| Regime_60_40       | 0.998 | 0.980 | 0.953 | 0.908 |
| Regime_Market      | 0.928 | 0.910 | 0.884 | 0.840 |

Sharpe ratio by half of the sample:

| index              | FirstHalfSharpe | SecondHalfSharpe |
| ------------------ | --------------- | ---------------- |
| BL_ML_Regime       | 1.076           | 0.549            |
| BL_ML_Regime_IC    | 1.123           | 0.677            |
| BL_ML_NoRegime     | 1.096           | 0.395            |
| BL_HistMean_Regime | 1.180           | 0.711            |
| Market_Reference   | 1.200           | 0.731            |
| Equal_Weight       | 0.856           | 0.733            |
| Static_60_40       | 1.473           | 0.663            |
| Inverse_Vol        | 1.002           | 0.573            |
| MeanVariance_LW    | 0.675           | 0.308            |
| Regime_60_40       | 1.488           | 0.660            |
| Regime_Market      | 1.171           | 0.728            |

Black-Litterman parameter sensitivity, one parameter changed at a time:

| Parameter           | Value     | Default | Variant         | Sharpe | MaxDrawdown | AnnTurnover | AvgCash |
| ------------------- | --------- | ------- | --------------- | ------ | ----------- | ----------- | ------- |
| tau                 | 0.025     |         | BL_ML_Regime    | 0.824  | -14.95%     | 3.35        | 16.3%   |
| tau                 | 0.025     |         | BL_ML_Regime_IC | 0.864  | -13.36%     | 1.60        | 17.7%   |
| tau                 | 0.05      | yes     | BL_ML_Regime    | 0.780  | -16.28%     | 4.37        | 16.8%   |
| tau                 | 0.05      | yes     | BL_ML_Regime_IC | 0.863  | -13.30%     | 1.92        | 19.0%   |
| tau                 | 0.1       |         | BL_ML_Regime    | 0.709  | -18.17%     | 5.45        | 17.1%   |
| tau                 | 0.1       |         | BL_ML_Regime_IC | 0.840  | -13.18%     | 2.34        | 21.2%   |
| tau                 | 0.2       |         | BL_ML_Regime    | 0.566  | -20.97%     | 6.28        | 17.0%   |
| tau                 | 0.2       |         | BL_ML_Regime_IC | 0.830  | -12.97%     | 2.81        | 24.6%   |
| base_delta          | 1.5       |         | BL_ML_Regime    | 0.727  | -18.60%     | 5.29        | 14.9%   |
| base_delta          | 1.5       |         | BL_ML_Regime_IC | 0.848  | -13.59%     | 2.28        | 18.5%   |
| base_delta          | 2.5       | yes     | BL_ML_Regime    | 0.780  | -16.28%     | 4.37        | 16.8%   |
| base_delta          | 2.5       | yes     | BL_ML_Regime_IC | 0.863  | -13.30%     | 1.92        | 19.0%   |
| base_delta          | 4.0       |         | BL_ML_Regime    | 0.827  | -14.83%     | 3.56        | 18.4%   |
| base_delta          | 4.0       |         | BL_ML_Regime_IC | 0.864  | -13.12%     | 1.65        | 19.5%   |
| base_delta          | 6.0       |         | BL_ML_Regime    | 0.855  | -14.03%     | 2.87        | 19.9%   |
| base_delta          | 6.0       |         | BL_ML_Regime_IC | 0.867  | -13.00%     | 1.47        | 19.9%   |
| stressed_delta_mult | 1.0       |         | BL_ML_Regime    | 0.801  | -22.41%     | 4.36        | 7.0%    |
| stressed_delta_mult | 1.0       |         | BL_ML_Regime_IC | 0.878  | -20.31%     | 1.22        | 5.9%    |
| stressed_delta_mult | 1.4       |         | BL_ML_Regime    | 0.787  | -18.52%     | 4.32        | 12.0%   |
| stressed_delta_mult | 1.4       |         | BL_ML_Regime_IC | 0.883  | -15.45%     | 1.61        | 14.0%   |
| stressed_delta_mult | 1.8       | yes     | BL_ML_Regime    | 0.780  | -16.28%     | 4.37        | 16.8%   |
| stressed_delta_mult | 1.8       | yes     | BL_ML_Regime_IC | 0.863  | -13.30%     | 1.92        | 19.0%   |
| stressed_delta_mult | 2.5       |         | BL_ML_Regime    | 0.756  | -14.19%     | 4.40        | 22.4%   |
| stressed_delta_mult | 2.5       |         | BL_ML_Regime_IC | 0.823  | -11.57%     | 2.23        | 23.9%   |
| stressed_omega_mult | 1.0       |         | BL_ML_Regime    | 0.672  | -17.97%     | 4.87        | 14.1%   |
| stressed_omega_mult | 1.0       |         | BL_ML_Regime_IC | 0.852  | -13.51%     | 1.99        | 18.4%   |
| stressed_omega_mult | 1.5       |         | BL_ML_Regime    | 0.728  | -17.04%     | 4.61        | 15.3%   |
| stressed_omega_mult | 1.5       |         | BL_ML_Regime_IC | 0.858  | -13.39%     | 1.94        | 18.7%   |
| stressed_omega_mult | 2.5       | yes     | BL_ML_Regime    | 0.780  | -16.28%     | 4.37        | 16.8%   |
| stressed_omega_mult | 2.5       | yes     | BL_ML_Regime_IC | 0.863  | -13.30%     | 1.92        | 19.0%   |
| stressed_omega_mult | 4.0       |         | BL_ML_Regime    | 0.804  | -15.72%     | 4.16        | 18.1%   |
| stressed_omega_mult | 4.0       |         | BL_ML_Regime_IC | 0.866  | -13.24%     | 1.91        | 19.1%   |
| max_weight          | 0.25      |         | BL_ML_Regime    | 0.722  | -15.86%     | 3.76        | 18.5%   |
| max_weight          | 0.25      |         | BL_ML_Regime_IC | 0.788  | -13.04%     | 1.79        | 20.9%   |
| max_weight          | 0.35      | yes     | BL_ML_Regime    | 0.780  | -16.28%     | 4.37        | 16.8%   |
| max_weight          | 0.35      | yes     | BL_ML_Regime_IC | 0.863  | -13.30%     | 1.92        | 19.0%   |
| max_weight          | 0.5       |         | BL_ML_Regime    | 0.778  | -16.48%     | 4.75        | 15.8%   |
| max_weight          | 0.5       |         | BL_ML_Regime_IC | 0.901  | -13.44%     | 2.04        | 17.9%   |
| prior               | reference | yes     | BL_ML_Regime    | 0.780  | -16.28%     | 4.37        | 16.8%   |
| prior               | reference | yes     | BL_ML_Regime_IC | 0.863  | -13.30%     | 1.92        | 19.0%   |
| prior               | equal     |         | BL_ML_Regime    | 0.643  | -18.37%     | 4.83        | 17.3%   |
| prior               | equal     |         | BL_ML_Regime_IC | 0.656  | -15.47%     | 2.05        | 18.0%   |

![Sharpe ratio across parameter settings](../results/figures/13_sensitivity.png)

_Sharpe ratio across parameter settings_

## 6. Discussion

The research hypothesis is not supported. The highest Sharpe ratio among the Kairos portfolios and the classical benchmarks belongs to Static_60_40. The Sharpe difference of BL_ML_Regime to the best classical benchmark (Static_60_40) is -0.20 (95% interval -0.53 to 0.07, p = 0.197), not statistically significant; BL_ML_Regime is significantly better than 1 and significantly worse than 0 of the 5 classical benchmarks (5% level). For BL_ML_Regime_IC the difference is -0.12 (95% interval -0.34 to 0.05, p = 0.233), not statistically significant.

The forecasting layer is the weak link: the stacked ensemble has a mean out-of-sample information coefficient of -0.037. 2 of 9 assets show an IC that is significantly negative at the 5% level (SPY, BTC-USD) and 0 show one that is significantly positive. The stacked ensemble beats the mean of the single models on 4 of 9 assets. Measured against an expanding historical-mean forecast, the stacked ensemble has a negative out-of-sample R2 on 9 of 9 assets (mean -0.170), and across all models and assets 53 of 54 model-asset R2 values are negative. Hit rates flatter the models: for SPY the stacked ensemble calls an up month in 83% of months and is right 59.7% of the time, while SPY rose in 68.7% of months, so an always-up rule would have scored higher. The stacked hit rate is below the always-up baseline on 8 of 9 assets. The LSTM is badly scaled: its forecasts vary about 1.9 times as much as realised returns, with a mean R2 of -3.8, which injects noise into the equal-weight ensemble.

One candidate explanation is that the first 48 months of training are dominated by 2008. For SPY, the rank correlation between the filtered stress probability and the next-month return is -0.23 in the initial training window and +0.21 afterwards, and the sign differs between the two windows for 6 of 9 assets. For SPY this is consistent with models learning a stress-precedes-losses rule from the crisis that the rebounds of later years then contradicted. Re-fitting the tree and linear models under 4 alternative designs on the dates common to all designs gives a mean IC across assets of -0.011 as specified, then no regime features -0.034; no stress features +0.018; longer initial window -0.008; no regime longer window -0.034. The largest improvement comes from the no stress features design (+0.028). The R2 against the historical mean stays negative under every design, so no design yields a forecast that beats the historical mean, and the weak skill is not simply an artefact of the regime features or the initial window. It points to a weak and unstable one-month return signal. Choosing a design after seeing out-of-sample results is a post hoc choice, so the pipeline keeps the specified design and treats this only as a diagnostic that needs fresh data to confirm.

The decisive ablation applies a mechanical version of the same stress rule (risky weights divided by 1.8, the remainder in cash) to benchmarks that use no forecasts. On static 60/40 it moves the Sharpe ratio from 0.98 to 0.98 (difference -0.001, p = 0.994) and the maximum drawdown from -20.5% to -13.1%. On the market reference the Sharpe ratio moves from 0.93 to 0.91 (p = 0.795) and the drawdown from -21.1% to -13.6%. The best overlay (Regime_60_40) has a Sharpe ratio of 0.98, against 0.78 for BL_ML_Regime and 0.86 for BL_ML_Regime_IC. Neither forecast-driven portfolio exceeds a rule that needs no forecasts (paired Sharpe differences -0.20 and -0.12, p = 0.23 and 0.20), so the machine learning and Black-Litterman machinery shows no measurable benefit beyond the defensive exposure cut. Black-Litterman with naive historical-mean views scores 0.89, and the Sharpe difference of BL_ML_Regime to it is -0.11 (95% interval -0.35 to 0.10, p = 0.333), not statistically significant, so the machine learning views do not improve on naive ones inside the same allocator.

The regime layer does reduce risk. BL_ML_Regime has a maximum drawdown of -16.3% against -22.8% without regime scaling, although the Sharpe gain is not significant. The cost shows in months flagged stressed at the decision date: BL_ML_Regime earned 8.3% annualised in those months, against 16.4% for Market_Reference and 17.7% for Equal_Weight. Stress flags in this sample were typically followed by rebounds, so cutting exposure protects capital in the drawdown but gives up return in the recovery.

The backtest starts in January 2009, after the crisis trough, so it covers a long equity bull market that favours benchmarks with a high equity weight and leaves the 2008 drawdown outside the test. The Sharpe ratio of static 60/40 is 1.47 in the first half of the sample and 0.66 in the second, and 11 of 11 strategies show a lower Sharpe ratio in the second half, so conclusions about relative performance depend on the period.

BL_ML_Regime_IC has the highest Sharpe ratio of the three Kairos variants, but it is the best of 3 variants tried on one sample and should be read as exploratory. Its deflated Sharpe ratio gives a probability of 0.999 that the true Sharpe ratio is positive after allowing for 3 trials and 0.998 after allowing for 10. This tests the Sharpe ratio against zero, not against 60/40, so it does not contradict the paired comparisons above. Against its sibling BL_ML_Regime the Sharpe difference is +0.08 (95% interval -0.11 to 0.29, p = 0.412), not statistically significant.

Across 42 one-at-a-time parameter changes (tau, base_delta, stressed_delta_mult, stressed_omega_mult, max_weight, prior), the Sharpe ratio ranges BL_ML_Regime from 0.57 to 0.86 and BL_ML_Regime_IC from 0.66 to 0.90, against 0.98 for Static_60_40, and 0 of 42 runs exceed the 60/40 value. No setting in the grid closes the gap, so the shortfall is not a tuning artefact. An equal-weight prior in place of the reference allocation gives 0.64 and 0.66.

## 7. Limitations

The usable out-of-sample window begins after the initial training period, so the 2008 crisis is not part of the portfolio backtest even though its regime labels are available once the HMM has enough history. Monthly data gives roughly 15 years of observations, which limits statistical power, so wide intervals should be read as inconclusive rather than as evidence of equivalence. The prior is a hand-picked reference allocation (SPY 42%, IEF 15%, EFA 12%, TLT 10%, GLD 8%, DBC 5%, HYG 5%, BTC-USD 2%, ETH-USD 1%), not an observed market-capitalisation series, and the equal-weight prior check in section 5.8 shows how far the results depend on it. The stress rule is calibrated by hand (multipliers fixed in advance) and the validation windows were chosen with hindsight, although they are not used in fitting. The LSTM is a small NumPy implementation refit every six months. The diagnostics are exploratory: they use the same sample as the main test. Yahoo adjusted closes are retrospective vendor data and can be revised. Results hold for this universe and period.

## 8. Reproducibility

Run `python run.py all` to download data, run every stage and rebuild this report. Seeds, parameters and paths are in `config.py`.

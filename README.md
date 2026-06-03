# Deep Learning-Based Trading Strategies for the Russian Stock Market

Master's thesis project focused on developing and evaluating machine learning and deep learning models for intraday trading in the Russian stock market.

The project investigates whether modern deep learning architectures can extract economically meaningful trading signals from noisy financial time series and outperform traditional machine learning approaches under realistic market conditions.

---

## Key Results

| Model                      | Asset      | Return |
| -------------------------- | ---------- | ------ |
| MLP                        | GLDRUB_TOM | 78.92% |
| BiLSTM + Attention         | YDEX       | 57.77% |
| 1D CNN + Wavelet Denoising | YDEX       | 53.42% |

### Main Findings

* Deep learning models consistently outperformed traditional machine learning approaches.
* Wavelet denoising significantly improved CNN performance on intraday market data.
* Adaptive volatility-based labeling improved class balance and reduced noise sensitivity.
* Macroeconomic features did not improve high-frequency trading performance.
* Statistical significance does not necessarily imply economic significance.
* Risk management and market regime filtering contributed more to robustness than marginal improvements in predictive accuracy.

---

## Research Motivation

Financial markets are characterized by:

* high noise levels;
* non-stationary behavior;
* changing market regimes;
* low signal-to-noise ratio.

Traditional machine learning models often achieve acceptable predictive metrics while failing to generate profitable trading strategies.

The goal of this project was to evaluate whether deep learning architectures can capture nonlinear temporal dependencies and generate statistically significant and economically meaningful trading signals.

---

## Dataset

### Market Data

* Source: Tinkoff Invest API
* Frequency: 5-minute candles
* Train Period: 2020–2023
* Test Period: 2024–2025
* Assets:

  * SBER
  * LKOH
  * YDEX
  * GLDRUB_TOM
  * additional liquid Russian securities

### Features

#### Market Features

* Open
* High
* Low
* Close
* Volume

#### Technical Indicators

* RSI
* MACD
* ATR
* Bollinger Bands
* Moving Averages

#### Context Features

* MOEX Index
* Relative Strength
* Market Volatility
* Key Interest Rate
* Inflation

---

## Adaptive Volatility Labeling

A custom dynamic labeling framework was developed.

Instead of using fixed thresholds, future returns are compared against local volatility estimates:

* Up
* Down
* Flat

This approach adapts class boundaries to current market conditions and improves robustness across different volatility regimes.

---

## Models Evaluated

### Traditional Machine Learning

* Ridge Regression
* Lasso Regression
* LightGBM
* CatBoost

### Deep Learning

#### Deep MLP

Fully connected neural network trained on engineered features.

#### Regular 1D CNN

Convolutional architecture designed to extract local temporal patterns from financial time series.

#### BiLSTM + Attention

Bidirectional recurrent architecture capable of capturing long-term dependencies and temporal context.

#### CDT 1D CNN

Cross-Data-Type CNN incorporating market context and macroeconomic information.

---

## Experimental Design

### Walk-Forward Validation

A realistic rolling evaluation framework was implemented:

* 48 months training window
* 1 month testing window
* incremental retraining

This prevents look-ahead bias and better approximates real trading conditions.

### Trading Assumptions

Transaction costs were explicitly modeled:

* Commission: 0.03%
* Slippage: 0.02%

All reported results include these costs.

---

## Noise Reduction

To reduce microstructure noise, wavelet denoising was applied using:

* Symlet 4 wavelet
* Level 3 decomposition

### Example Impact

YDEX (CNN):

* Without denoising: 10.96%
  return
* With denoising: 53.42%
  return

This suggests that a significant portion of intraday price fluctuations represents noise rather than useful information.

---

## Risk Management

The project includes a dynamic breakeven mechanism.

Features:

* Adaptive stop-loss management
* Breakeven protection
* Position exit logic
* Volatility-aware risk control

Results:

* Win Rate improved from 37.64% to 52.10%
* Reduction in large losing trades
* Improved equity curve stability

---

## Statistical Significance

Performance was evaluated using:

* Total Return
* Sharpe Ratio
* Profit Factor
* Maximum Drawdown
* Win Rate
* Trade Count

In addition, statistical significance was assessed through Minimal Detectable Effect (MDE) analysis to distinguish genuine predictive signals from random outcomes.

---

## Key Conclusions

1. Traditional machine learning models failed to consistently extract profitable signals from 5-minute Russian market data.

2. Deep learning architectures demonstrated superior ability to model nonlinear market behavior.

3. Wavelet denoising produced substantial improvements for CNN-based strategies.

4. Macroeconomic variables provided limited value for high-frequency trading models.

5. The primary practical value of deep learning lies in adaptive risk management and market regime filtering rather than serving as a standalone source of alpha.

---

## Technologies Used

* Python
* PyTorch
* Scikit-Learn
* Pandas
* NumPy
* LightGBM
* CatBoost
* Optuna
* PyWavelets
* Matplotlib
* Seaborn

---

## Repository Structure

```text
.
├── notebooks/
├── results/
├── thesis/
├── presentation/
└── README.md
```

---

## Future Work

Potential directions for further research:

* Transformer-based architectures
* Market regime detection models
* Reinforcement learning approaches
* Multi-asset portfolio optimization
* News and sentiment integration
* Cross-market transfer learning

---

## Author

Alexey Smirnov

Master's Thesis in Data Analysis / Quantitative Finance

2026

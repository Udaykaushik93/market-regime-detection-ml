# Hybrid ML Market Regime Detection (NIFTY + VIX)

This project builds a hybrid machine learning framework to detect market regimes
and forecast market risk using NIFTY and India VIX data.

The model combines unsupervised clustering with supervised ML to identify
hidden market states and predict future market behavior.

---

## Features

- Market regime detection using Gaussian Mixture Models
- Crash probability prediction (10-day & 30-day)
- Expected return forecasting
- Volatility stress modeling
- Market structure instability detection
- Regime transition probability analysis

---

## Machine Learning Models

Unsupervised Learning
- Gaussian Mixture Model (Market Regimes)

Supervised Learning
- LightGBM Classifier (Crash Prediction)
- LightGBM Regressor (Return Prediction)

---

## Feature Engineering

Key features engineered:

- Trend Strength
- Volatility Stress
- Structure Stress
- Direction Bias Score
- VIX Z-score
- VIX acceleration
- Slope decay

---

## Data

Source:
- NIFTY Index
- India VIX

Data collected using:

yfinance

---

## Output

The model generates a market state vector including:

- Crash probability (10d / 30d)
- Expected return
- Market trend state
- Volatility regime
- Structural stability
- Market direction bias

---

## Use Cases

- Quantitative trading research
- Risk management
- Market regime detection
- Portfolio allocation strategies

---

## Author

Uday Kaushik

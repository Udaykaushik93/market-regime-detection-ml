# Market Regime Detection using Machine Learning

This project implements a machine learning framework for analyzing financial markets and detecting different market regimes such as expansion, pullback, correction, and crash.

## Overview

The system uses financial time-series data and machine learning models to identify structural stress, volatility conditions, and market direction.

The model combines:

- Feature engineering on price and volatility signals
- Gaussian Mixture Models for unsupervised regime detection
- LightGBM models for regime classification and return prediction
- Market state interpretation for risk assessment

## Technologies Used

- Python
- Pandas
- NumPy
- LightGBM
- Scikit-learn
- yfinance

## Data Source

Market data is retrieved from Yahoo Finance using the `yfinance` API.

## Project Structure


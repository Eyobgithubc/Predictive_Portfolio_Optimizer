# data_analysis_functions.py
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler

def download_data(tickers, start_date, end_date):
    """Download adjusted close price data for specified tickers and date range."""
    data = yf.download(tickers, start=start_date, end=end_date)
    data = data['Adj Close']
    data.columns = tickers
    return data

def clean_data(data):
    """Handle missing values by forward filling and then dropping any remaining NaNs."""
    data.fillna(method='ffill', inplace=True)
    data.dropna(inplace=True)
    return data

def normalize_data(data):
    """Normalize data using StandardScaler."""
    scaler = StandardScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)
    return scaled_data

def plot_closing_prices(data):
    """Plot the closing prices of each ticker over time."""
    plt.figure(figsize=(14, 7))
    for col in data.columns:
        plt.plot(data[col], label=col)
    plt.title('Closing Prices')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Closing Price')
    plt.legend()
    plt.show()

def calculate_daily_returns(data):
    """Calculate daily returns for each ticker."""
    returns = data.pct_change().dropna()
    return returns

def plot_daily_returns(returns):
    """Plot daily returns of each ticker."""
    returns.plot(figsize=(14, 7))
    plt.title('Daily Returns')
    plt.xlabel('Date')
    plt.ylabel('Daily Return')
    plt.show()

def plot_rolling_statistics(data, ticker, window=30):
    """Plot rolling mean and standard deviation for a specific ticker."""
    rolling_mean = data[ticker].rolling(window=window).mean()
    rolling_std = data[ticker].rolling(window=window).std()

    plt.figure(figsize=(14, 7))
    plt.plot(data[ticker], label=f'{ticker} Original')
    plt.plot(rolling_mean, label=f'{ticker} {window}-Day Rolling Mean', color='orange')
    plt.plot(rolling_std, label=f'{ticker} {window}-Day Rolling Std Dev', color='red')
    plt.title(f'{ticker} Stock Price with Rolling Mean and Std Dev')
    plt.legend()
    plt.show()

def detect_outliers(returns):
    """Detect outliers in daily returns, defined as returns beyond 3 standard deviations."""
    outliers = returns[(np.abs(returns - returns.mean()) > 3 * returns.std())]
    return outliers.dropna(how='all')

def plot_outliers(returns, outliers, ticker):
    """Plot daily returns with outliers highlighted for a specific ticker."""
    plt.figure(figsize=(14, 7))
    plt.plot(returns[ticker], label='Daily Returns')
    plt.scatter(outliers.index, outliers[ticker], color='red', label='Outliers')
    plt.title(f'{ticker} Daily Returns with Outliers')
    plt.xlabel('Date')
    plt.ylabel('Daily Return')
    plt.legend()
    plt.show()

def decompose_time_series(data, ticker, period=252):
    """Decompose time series for a given ticker to observe trend, seasonality, and residuals."""
    decomposition = seasonal_decompose(data[ticker], model='multiplicative', period=period)
    decomposition.plot()
    plt.show()

def calculate_risk_metrics(returns, ticker, risk_free_rate=0.01):
    """Calculate Value at Risk (VaR) and Sharpe Ratio for a specific ticker."""
    var_95 = returns[ticker].quantile(0.05)
    sharpe_ratio = (returns[ticker].mean() - risk_free_rate) / returns[ticker].std()
    return var_95, sharpe_ratio

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt


def create_portfolio_dataframe(tsla_forecast, bnd_forecast, spy_forecast):
    df = pd.DataFrame({
        'TSLA': tsla_forecast['Close'],
        'BND': bnd_forecast['Close'],
        'SPY': spy_forecast['Close']
    })
    return df


def calculate_annual_returns(df):
    daily_returns = df.pct_change().dropna()
    annual_returns = daily_returns.mean() * 252
    return annual_returns, daily_returns


def calculate_covariance_matrix(daily_returns):
    return daily_returns.cov() * 252  


def calculate_portfolio_performance(weights, annual_returns, cov_matrix):
    portfolio_return = np.dot(weights, annual_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_volatility


def negative_sharpe(weights, annual_returns, cov_matrix, risk_free_rate=0.02):
    port_return, port_volatility = calculate_portfolio_performance(weights, annual_returns, cov_matrix)
    sharpe_ratio = (port_return - risk_free_rate) / port_volatility
    return -sharpe_ratio


def optimize_portfolio(annual_returns, cov_matrix, initial_weights, risk_free_rate=0.02):
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  
    bounds = tuple((0, 1) for _ in range(len(initial_weights)))

    optimized = minimize(
        negative_sharpe, initial_weights,
        args=(annual_returns, cov_matrix, risk_free_rate),
        bounds=bounds, constraints=constraints
    )
    return optimized.x  


def calculate_var(daily_returns, confidence_level=0.95):
    tsla_mean_return = daily_returns['TSLA'].mean()
    tsla_std_dev = daily_returns['TSLA'].std()
    VaR_TSLA = norm.ppf(1 - confidence_level) * tsla_std_dev - tsla_mean_return
    return VaR_TSLA


def plot_cumulative_returns(daily_returns, optimal_weights):
    cumulative_returns = (1 + daily_returns).cumprod()
    portfolio_cumulative_return = cumulative_returns.dot(optimal_weights)

    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_cumulative_return, label="Optimized Portfolio")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.title("Portfolio Performance")
    plt.legend()
    plt.show()


def summarize_portfolio_performance(optimal_weights, annual_returns, cov_matrix, risk_free_rate=0.02):
    portfolio_return, portfolio_volatility = calculate_portfolio_performance(optimal_weights, annual_returns, cov_matrix)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

    summary = {
        "Expected Annual Return": portfolio_return,
        "Annualized Volatility": portfolio_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Optimal Weights": optimal_weights
    }
    return summary

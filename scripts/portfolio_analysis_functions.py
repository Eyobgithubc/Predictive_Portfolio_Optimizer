import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

def calculate_portfolio_return(weights, returns):
 
    return np.sum(returns.mean() * weights) * 252

def calculate_portfolio_volatility(weights, returns):
 
    return np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))

def portfolio_statistics(weights, returns, risk_free_rate=0.01):
    
    port_return = calculate_portfolio_return(weights, returns)
    port_volatility = calculate_portfolio_volatility(weights, returns)
    sharpe_ratio = (port_return - risk_free_rate) / port_volatility
    return port_return, port_volatility, sharpe_ratio

def generate_random_portfolios(num_portfolios, returns, risk_free_rate=0.01):
   
    num_assets = len(returns.columns)
    results = np.zeros((3, num_portfolios))
    weight_matrix = np.zeros((num_portfolios, num_assets))

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weight_matrix[i, :] = weights
        port_return, port_volatility, sharpe_ratio = portfolio_statistics(weights, returns, risk_free_rate)
        results[0, i] = port_return
        results[1, i] = port_volatility
        results[2, i] = sharpe_ratio

    return results, weight_matrix

def plot_efficient_frontier(results):
   
    plt.figure(figsize=(10, 6))
    plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis')
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.title('Efficient Frontier')
    plt.show()

def optimize_portfolio(returns, risk_free_rate=0.01):
  
    num_assets = len(returns.columns)
    args = (returns, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))

    def neg_sharpe_ratio(weights, returns, risk_free_rate):
      
        return -portfolio_statistics(weights, returns, risk_free_rate)[2]

    initial_guess = num_assets * [1. / num_assets]
    result = minimize(neg_sharpe_ratio, initial_guess, args=args, method='SLSQP', bounds=bounds, constraints=constraints)

    return result.x

def display_optimal_portfolio(weights, returns):
   
    port_return, port_volatility, sharpe_ratio = portfolio_statistics(weights, returns)
    print(f"Optimal Portfolio Return: {port_return:.2f}")
    print(f"Optimal Portfolio Volatility: {port_volatility:.2f}")
    print(f"Optimal Portfolio Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Optimal Weights: {weights}")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# Assuming the CSV file is properly loaded and columns match as described
indices = ['IonQ', 'Rigetti Computing', 'Quantum Computing Inc.',
           'D-Wave Quantum', 'Alphabet', 'IBM', 
           'Microsoft', 'Nvidia', 'Defiance Quantum ETF', 'Global X Future Analytics Tech']
colnames = ['Date','IONQ','RGTI','QUBT','QBTS','GOOGL','IBM','MSFT','NVDA','QTUM','AIQ']
symbols = ['IONQ','RGTI','QUBT','QBTS','GOOGL','IBM','MSFT','NVDA','QTUM','AIQ']

# Load and preprocess data
df = pd.read_csv('quantum_technology_indices_prices.csv')
df = df.dropna(axis=0)
df.columns = colnames
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Load and preprocess data
df = pd.read_csv('quantum_technology_indices_prices.csv')
df = df.dropna(axis=0)
df.columns = colnames
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Compute daily returns
returns = df[symbols].pct_change().dropna()

# Calculate mean returns and covariance matrix
mean_returns = returns.mean()
cov_matrix = returns.cov()
risk_free_rate = 0.05 / 252  # Daily risk-free rate (assuming 5% annualized)

# Assume market portfolio (can use actual market data if available)
market_return = returns.mean().mean()
market_std = returns.std().mean()
market_beta = 1.0  # Benchmark market beta

# Generate random portfolios
num_portfolios = 10000
results = np.zeros((6, num_portfolios))  # Extra rows for Treynor and Jensen
weights_record = []

for i in range(num_portfolios):
    # Generate random weights
    weights = np.random.random(len(symbols))
    weights /= np.sum(weights)
    weights_record.append(weights)

    # Portfolio return and variance
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Portfolio beta (weighted average of betas, assuming beta = 1 for simplicity)
    portfolio_beta = np.sum(weights)

    # Downside risk (Sortino Ratio)
    downside_std = np.sqrt(np.mean(np.minimum(returns.dot(weights), 0) ** 2))

    # Calculate metrics
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_stddev
    treynor_ratio = (portfolio_return - risk_free_rate) / portfolio_beta
    jensens_alpha = portfolio_return - (risk_free_rate + portfolio_beta * (market_return - risk_free_rate))
    sortino_ratio = (portfolio_return - risk_free_rate) / downside_std

    # Store results
    results[0, i] = portfolio_return
    results[1, i] = portfolio_stddev
    results[2, i] = sharpe_ratio
    results[3, i] = treynor_ratio
    results[4, i] = jensens_alpha
    results[5, i] = sortino_ratio

# Convert results to DataFrame
results_df = pd.DataFrame({
    "Return": results[0],
    "Volatility": results[1],
    "Sharpe Ratio": results[2],
    "Treynor Ratio": results[3],
    "Jensen's Alpha": results[4],
    "Sortino Ratio": results[5],
})
weights_df = pd.DataFrame(weights_record, columns=symbols)

# Find portfolios with max/min for different metrics
max_sharpe_idx = results_df["Sharpe Ratio"].idxmax()
max_treynor_idx = results_df["Treynor Ratio"].idxmax()
max_jensen_idx = results_df["Jensen's Alpha"].idxmax()

# Subplots for all metrics
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Sharpe Ratio
axes[0, 0].scatter(results_df["Volatility"], results_df["Return"], c=results_df["Sharpe Ratio"], 
    cmap='viridis', marker='o', s = 5, alpha = 0.5)
axes[0, 0].set_title("Sharpe Ratio", fontsize = 12, weight = 'bold')
axes[0, 0].set_xlabel("Volatility", fontsize = 12, weight = 'bold')
axes[0, 0].set_ylabel("Return", fontsize = 12, weight = 'bold')

# Treynor Ratio
axes[0, 1].scatter(results_df["Volatility"], results_df["Return"], c=results_df["Treynor Ratio"], 
    cmap='viridis', marker='o', s = 5, alpha = 0.5)
axes[0, 1].set_title("Treynor Ratio", fontsize = 12, weight = 'bold')
axes[0, 1].set_xlabel("Volatility", fontsize = 12, weight = 'bold')
axes[0, 1].set_ylabel("Return", fontsize = 12, weight = 'bold')

# Jensen's Alpha
axes[1, 0].scatter(results_df["Volatility"], results_df["Return"], c=results_df["Jensen's Alpha"], 
    cmap='viridis', marker='o', s = 5, alpha = 0.5)
axes[1, 0].set_title("Jensen's Alpha", fontsize = 12, weight = 'bold')
axes[1, 0].set_xlabel("Volatility", fontsize = 12, weight = 'bold')
axes[1, 0].set_ylabel("Return", fontsize = 12, weight = 'bold')

# Sortino Ratio
sortino_ratios = (results[0] - risk_free_rate) / results[1]  # Pre-computed downside
axes[1, 1].scatter(results_df["Volatility"], results_df["Return"], c=results_df["Sortino Ratio"], 
    cmap='viridis', marker='o', s = 5, alpha = 0.5)
axes[1, 1].set_title("Sortino Ratio", fontsize = 12, weight = 'bold')
axes[1, 1].set_xlabel("Volatility", fontsize = 12, weight = 'bold')
axes[1, 1].set_ylabel("Return", fontsize = 12, weight = 'bold')

plt.tight_layout()
plt.savefig('eff_frontier.jpg', dpi = 600)
plt.show()

weights_df = pd.DataFrame(weights_record, columns=symbols)
print(weights_df)

fig, axes = plt.subplots(2,2,figsize=(12, 6))
axes = axes.flatten()

metrics = ["Sharpe Ratio", "Treynor Ratio", "Jensen's Alpha", "Sortino Ratio"]
for i, metric  in enumerate(metrics):
    # Find the best portfolio based on Sharpe Ratio (or any other metric)
    best_metric = metric  # Change this to "Treynor Ratio", "Jensen's Alpha", or "Sortino Ratio" as needed
    best_idx = i

    # Extract the best portfolio weights and metrics
    best_portfolio_weights = weights_df.iloc[best_idx]
    best_portfolio_return = results_df["Return"][best_idx]
    best_portfolio_volatility = results_df["Volatility"][best_idx]

    # Print best portfolio details
    print('-----------------')
    print(f"Best Portfolio based on {best_metric}:\n")
    #print(f"Portfolio Weights:\n{best_portfolio_weights}")
    print(f"\nExpected Annualized Return: {best_portfolio_return * 252:.2%}")
    print(f"Expected Annualized Volatility: {best_portfolio_volatility * np.sqrt(252):.2%}")

    # Visualize the expected return of the best portfolio

    axes[i].bar(best_portfolio_weights.index, best_portfolio_weights, color='blue', label = metric, alpha = 0.8)
    axes[i].set_title(f"{best_metric}", fontsize = 12, weight = 'bold')
    axes[i].set_xlabel("Asset", fontsize = 12, weight = 'bold')
    axes[i].set_ylabel("Weight", fontsize = 12, weight = 'bold')
    axes[i].set_xticks(symbols, rotation=90 , fontsize = 12, weight = 'bold')
    #axes[i].grid(True)
plt.tight_layout()
plt.savefig('portopt_weight.jpg', dpi = 600)
plt.show()
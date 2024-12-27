import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Parameters for the model
T = 200  # Number of time points
V_H = 105  # High fundamental value
V_L = 95   # Low fundamental value
lambda_values = np.linspace(0.1, 0.9, 9)  # Different probabilities of informed trader

# Generate the true value of the asset (mu) as a random walk
mu = np.cumsum(np.random.normal(0, 1, T)) + 100  # Random walk centered at 100

# Store results in a DataFrame
df = pd.DataFrame({'Time': np.arange(T), 'True Value (mu)': mu})

# Function to compute bid, ask, and spread for a given lambda
def compute_bid_ask_spread(V_H, V_L, mu, lam):
    """
    Computes the bid, ask, and spread based on Glosten-Milgrom model.
    
    Parameters:
    V_H (float): High fundamental value
    V_L (float): Low fundamental value
    mu (array): True value of the asset at each time
    lam (float): Probability of informed trader
    
    Returns:
    dict: Bid, Ask, and Spread arrays
    """
    S_t = 2 * lam * (V_H - V_L)  # Spread at each time
    bid = mu - S_t / 2  # Bid price
    ask = mu + S_t / 2  # Ask price
    return {'Bid': bid, 'Ask': ask, 'Spread': S_t}

# Generate bid, ask, and spread for different lambda values
results = {}

for lam in lambda_values:
    bid_ask_spread = compute_bid_ask_spread(V_H, V_L, mu, lam)
    results[lam] = pd.DataFrame({
        'Time': np.arange(T),
        'Bid': bid_ask_spread['Bid'],
        'Ask': bid_ask_spread['Ask'],
        'Spread': bid_ask_spread['Spread'],
        'Lambda': lam
    })

# Combine all results into a single DataFrame for visualization
combined_df = pd.concat(results.values(), ignore_index=True)

# ---------------------------------------------
# PLOT 1: True Value (mu) vs Bid, Ask, Spread
# ---------------------------------------------
plt.figure(figsize=(12, 6))
for lam in [0.1, 0.3, 0.5, 0.7, 0.9]:
    subset = combined_df[combined_df['Lambda'] == lam]
    plt.plot(subset['Time'], subset['Spread'], label=f'λ = {lam:.1f}')
    
plt.title('Bid-Ask Spread Over Time for Different λ (Informed Trader Probabilities)')
plt.xlabel('Time')
plt.ylabel('Spread ($S_t$)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# ---------------------------------------------
# PLOT 2: Bid, Ask, and True Value (for fixed λ)
# ---------------------------------------------
lambda_example = 0.5  # Select a specific lambda to visualize
subset = combined_df[combined_df['Lambda'] == lambda_example]

plt.figure(figsize=(12, 6))
plt.plot(subset['Time'], subset['Bid'], label='Bid', color='red', linestyle='--')
plt.plot(subset['Time'], subset['Ask'], label='Ask', color='green', linestyle='--')
plt.plot(subset['Time'], df['True Value (mu)'], label='True Value (μ)', color='blue')
plt.fill_between(subset['Time'], subset['Bid'], subset['Ask'], color='orange', alpha=0.2, label='Spread ($S_t$)')

plt.title(f'Bid, Ask, and True Value for λ = {lambda_example:.2f}')
plt.xlabel('Time')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# ---------------------------------------------
# PLOT 3: Heatmap of Spread as a function of λ
# ---------------------------------------------
# Pivot the combined DataFrame to create a heatmap of Spread vs Time for each lambda
heatmap_data = combined_df.pivot(index='Time', columns='Lambda', values='Spread')

plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, cmap='viridis', cbar=True, linewidths=0.1)
plt.title('Heatmap of Spread ($S_t$) as a Function of λ and Time')
plt.xlabel('Lambda (Probability of Informed Trader)')
plt.ylabel('Time')
plt.show()

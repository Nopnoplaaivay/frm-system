import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import welch


# Assuming the CSV file is properly loaded and columns match as described
indices = ['IonQ', 'Rigetti Computing', 'Quantum Computing Inc.',
           'D-Wave Quantum', 'Alphabet', 'IBM', 
           'Microsoft', 'Nvidia', 'Defiance Quantum ETF', 'Global X Future Analytics Tech']
colnames = ['Date','IONQ','RGTI','QUBT','QBTS','GOOGL','IBM','MSFT','NVDA','QTUM','AIQ']
symbols = ['IONQ','RGTI','QUBT','QBTS','GOOGL','IBM','MSFT','NVDA','QTUM','AIQ']
df = pd.read_csv('quantum_technology_indices_prices.csv')
df = df.dropna(axis=0)
df.columns = colnames
#df['Date'] = pd.to_datetime(df['Date'])  # Ensure the Date column is in datetime format

# Autocorrelation and Autocovariance functions
def autocorrelation(series, lag):
    return series.autocorr(lag)

def autocovariance(series, lag):
    mean = series.mean()
    return np.mean((series[:-lag] - mean) * (series[lag:] - mean)) if lag > 0 else np.var(series)

# Initialize dictionaries to store autocorrelation and autocovariance
autocorr_results = {symbol: [] for symbol in symbols}
autocov_results = {symbol: [] for symbol in symbols}
max_lag = 30  # Max lag for autocorrelation and autocovariance

# Compute for each symbol
for symbol in symbols:
    series = df[symbol]
    for lag in range(1, max_lag + 1):  # Start from lag 1
        autocorr_results[symbol].append(autocorrelation(series, lag))
        autocov_results[symbol].append(autocovariance(series.values, lag))

# Convert results to DataFrames for easier visualization
autocorr_df = pd.DataFrame(autocorr_results, index=range(1, max_lag + 1))
autocov_df = pd.DataFrame(autocov_results, index=range(1, max_lag + 1))

# Visualization


# Create a single figure with 4 subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Subplot 1: Autocorrelation Line Plot
for symbol in symbols:  # Visualize the first 5 stocks
    axs[0, 0].plot(autocorr_df.index, autocorr_df[symbol], label=f'{symbol}', marker = 'o', alpha = 0.5, lw = 1)
#axs[0, 0].set_title('Autocorrelation')
axs[0, 0].set_ylabel('Autocorrelation', fontsize = 12, weight = 'bold')
axs[0, 0].set_xlabel('Lag', fontsize = 12, weight = 'bold')
axs[0, 0].legend()
axs[0, 0].grid(True)

# Subplot 2: Autocovariance Line Plot
for symbol in symbols:  # Visualize the first 5 stocks
    axs[0, 1].plot(autocov_df.index, autocov_df[symbol], label=f'{symbol}', marker = 'o', alpha = 0.5, lw = 1)
#axs[0, 1].set_title('Autocovariance')
axs[0, 1].set_ylabel('Autocovariance', fontsize = 12, weight = 'bold')
axs[0, 1].set_xlabel('Lag', fontsize = 12, weight = 'bold')
axs[0, 1].grid(True)

# Subplot 3: Autocorrelation Heatmap
sns.heatmap(autocorr_df.T, cmap="coolwarm", annot=False, ax=axs[1, 0])
axs[1, 0].set_title('Autocorrelation Heatmap', fontsize = 12, weight = 'bold')
axs[1, 0].set_xlabel('Lag', fontsize = 12, weight = 'bold')
axs[1, 0].set_ylabel('Symbols', fontsize = 12, weight = 'bold')

# Subplot 4: Autocovariance Heatmap
sns.heatmap(autocov_df.T, cmap="coolwarm", annot=False, ax=axs[1, 1])
axs[1, 1].set_title('Autocovariance Heatmap', fontsize = 12, weight = 'bold')
axs[1, 1].set_xlabel('Lag', fontsize = 12, weight = 'bold')
axs[1, 1].set_ylabel('Symbols', fontsize = 12, weight = 'bold')


# Adjust layout
plt.tight_layout()
plt.savefig('autocorr.jpg', dpi = 600)
plt.show()

df = df[symbols]
print(df)


# Calculate log returns
log_returns = np.log(df / df.shift(1)).dropna()

# Define a function to estimate spectral density
def compute_spectral_density(data, symbol, fs=1.0):
    freqs, psd = welch(data, fs=fs, nperseg=min(256, len(data)))  # Welch's method
    return freqs, psd

# Plot spectral density for selected stocks
plt.figure(figsize=(10, 4))

for i, symbol in enumerate(symbols):  # Analyze the first 5 stocks
    freqs, psd = compute_spectral_density(log_returns[symbol], symbol)
    plt.plot(freqs, psd, label=symbol)

plt.title('Spectral Density of Log Returns', fontsize = 12, weight = 'bold')
plt.xlabel('Frequency', fontsize = 12, weight = 'bold')
plt.ylabel('Power Spectral Density', fontsize = 12, weight = 'bold')
plt.yscale('log')  # Log scale for better visualization of spectral features
plt.legend(ncol = 5)
plt.grid(True)
plt.tight_layout()
plt.savefig('power_spectral_density.jpg', dpi = 600)
plt.show()

# Example: Dominant frequencies
dominant_frequencies = {symbol: freqs[np.argmax(psd)] for symbol, (freqs, psd) in 
                         {sym: compute_spectral_density(log_returns[sym], sym) for sym in symbols}.items()}
print("Dominant Frequencies:")
print(dominant_frequencies)

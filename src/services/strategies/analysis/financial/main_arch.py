import pandas as pd
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt

vol_model = 'Garch'

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

# Compute daily returns for each symbol
returns = df[symbols].pct_change().dropna()


# Initialize results dictionary to store models
results = {}

# Set up subplots
num_symbols = len(symbols)
fig, axes = plt.subplots(nrows=(num_symbols + 1) // 2, ncols=2, figsize=(12, 10))
axes = axes.flatten()  # Flatten axes for easy iteration

# Fit GARCH model and plot conditional volatility
for i, symbol in enumerate(symbols):
    print(f"Processing {symbol}...")
    model = arch_model(returns[symbol]*1000, vol=vol_model, p=1, q=1, 
        mean='Constant', dist='Normal')
    res = model.fit(disp='off')
    results[symbol] = res

    # Plot conditional volatility on the subplot
    res.conditional_volatility.plot(ax=axes[i], color = 'red')
    axes[i].set_ylabel(r'$\sigma \times 10^3$', fontsize = 14, weight = 'bold')
    axes[i].set_title(label=f'{symbol}', fontsize=16, weight = 'bold')
    #axes[i].set_xlabel('Date', fontsize = 12, weight = 'bold')
    axes[i].grid(True)

# Remove extra subplots if the number of symbols is odd
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Adjust layout
plt.tight_layout()
plt.savefig(f'{vol_model}.jpg', dpi = 600)
plt.show()




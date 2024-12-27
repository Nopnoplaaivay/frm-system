import yfinance as yf
import pandas as pd
from datetime import datetime

# List of indices with their symbols
indices = [
    {"name": "IonQ", "symbol": "IONQ", "exchange": "NYSE"},
    {"name": "Rigetti Computing", "symbol": "RGTI", "exchange": "NASDAQ"},
    {"name": "Quantum Computing Inc.", "symbol": "QUBT", "exchange": "NASDAQ"},
    {"name": "D-Wave Quantum", "symbol": "QBTS", "exchange": "NYSE"},
    {"name": "Alphabet", "symbol": "GOOGL", "exchange": "NASDAQ"},
    {"name": "IBM", "symbol": "IBM", "exchange": "NYSE"},
    {"name": "Microsoft", "symbol": "MSFT", "exchange": "NASDAQ"},
    {"name": "Nvidia", "symbol": "NVDA", "exchange": "NASDAQ"},
    {"name": "Defiance Quantum ETF", "symbol": "QTUM", "exchange": "NYSEARCA"},
    {"name": "Global X Future Analytics Tech", "symbol": "AIQ", "exchange": "NASDAQ"},
    # "BlueStar QC and ML Index" symbol ("BQTUM") and exchange are unknown
]

# Define date range
start_date = "2019-01-01"
end_date = datetime.now().strftime("%Y-%m-%d")

# Create a dictionary to store data
stock_data = {}

# Download data for each index
for index in indices:
    try:
        ticker = yf.Ticker(index["symbol"])
        data = ticker.history(start=start_date, end=end_date)
        stock_data[index["name"]] = data
        print(f"Downloaded data for {index['name']} ({index['symbol']})")
    except Exception as e:
        print(f"Failed to download data for {index['name']} ({index['symbol']}): {e}")

# Combine all data into a single DataFrame for better analysis
combined_data = pd.concat(
    {name: data["Close"] for name, data in stock_data.items()},
    axis=1
)

# Save to CSV for later use
combined_data.to_csv("quantum_technology_indices_prices.csv")
print("Saved stock price data to quantum_technology_indices_prices.csv")

# Display the first few rows of combined data
print(combined_data.head())

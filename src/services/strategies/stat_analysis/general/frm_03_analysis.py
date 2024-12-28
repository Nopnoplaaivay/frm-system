import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming the CSV file is properly loaded and columns match as described
indices = ['IonQ', 'Rigetti Computing', 'Quantum Computing Inc.',
           'D-Wave Quantum', 'Alphabet', 'IBM', 
           'Microsoft', 'Nvidia', 'Defiance Quantum ETF', 'Global X Future Analytics Tech']

symbols = ['Date','IONQ','RGTI','QUBT','QBTS','GOOGL','IBM','MSFT','NVDA','QTUM','AIQ']

df = pd.read_csv('quantum_technology_indices_prices.csv')
df = df.dropna(axis=0)
df.columns = symbols
df = df.reset_index(inplace = False)
df = df.drop(columns = ['index'])
symbols = ['IONQ','RGTI','QUBT','QBTS','GOOGL','IBM','MSFT','NVDA','QTUM','AIQ']
print(df)
print(df.columns)


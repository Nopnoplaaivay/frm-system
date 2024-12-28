import optuna
import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import matplotlib.pyplot as plt



# Assuming the CSV file is properly loaded and columns match as described
indices = ['IonQ', 'Rigetti Computing', 'Quantum Computing Inc.',
           'D-Wave Quantum', 'Alphabet', 'IBM', 
           'Microsoft', 'Nvidia', 'Defiance Quantum ETF', 'Global X Future Analytics Tech']
colnames = ['Date','IONQ','RGTI','QUBT','QBTS','GOOGL','IBM','MSFT','NVDA','QTUM','AIQ']
symbols = ['IONQ','RGTI','QUBT','QBTS','GOOGL','IBM','MSFT','NVDA','QTUM','AIQ']
df = pd.read_csv('quantum_technology_indices_prices.csv')
df = df.dropna(axis=0)
df.columns = colnames
df['Date'] = pd.to_datetime(df['Date'])  # Ensure the Date column is in datetime format
df.set_index('Date', inplace=True)

lag = 5

print(df)
print(df.columns)


log_DF = pd.DataFrame([])

for index in range(len(symbols)):
    symbol = symbols[index]
    prices = df[symbol]
    # Calculate log returns
    log_returns = np.log(prices / prices.shift(lag)).dropna()
    log_returns_mean = log_returns.mean()
    log_returns_std = log_returns.std()
    log_returns = (log_returns - log_returns_mean)/(log_returns_std)
    #log_returns.index = pd.RangeIndex(start=0, stop=len(log_returns), step=1)
    log_DF[symbol] = log_returns


print(log_DF)

# Step 1: Standardize the data (optional, based on requirement)
standardized_df = (log_DF - log_DF.mean()) / log_DF.std()

# Step 2: Compute pairwise distances
# Use pdist for pairwise distance and squareform to convert to a matrix
distance_matrix = squareform(pdist(standardized_df.T, metric='euclidean'))

# Create a DataFrame for the distance matrix
distance_df = pd.DataFrame(distance_matrix, index=df.columns, columns=df.columns)

# Step 3: Visualize the pairwise distance matrix
plt.figure(figsize=(6, 6))
# Step 1: Create a mask for the upper triangle
sns.heatmap(distance_df, annot=True, cmap="coolwarm", fmt=".1f")
#plt.title("Pairwise Euclidean Distance Between Stocks")
#plt.xlabel("Stocks", rotation  = 0, weight = 'bold', fontsize = 16)
#plt.ylabel("Stocks", rotation  = 90, weight = 'bold', fontsize = 16)
plt.xticks(fontsize = 16, weight = 'bold', rotation  = 90)
plt.yticks(fontsize = 16, weight = 'bold', rotation  = 0)
plt.tight_layout()
plt.savefig('distance_01.jpg', dpi = 600)
plt.show()

import networkx as nx
from scipy.cluster.hierarchy import dendrogram, linkage

# Step 1: Compute the MST
# Convert distance DataFrame to a format usable by networkx
mst_graph = nx.Graph()

# Add edges with weights based on the distance matrix
for i, stock1 in enumerate(distance_df.columns):
    for j, stock2 in enumerate(distance_df.columns):
        if i < j:  # Avoid duplicate edges
            mst_graph.add_edge(stock1, stock2, weight=distance_df.iloc[i, j])

# Compute the MST using Kruskal's algorithm
mst = nx.minimum_spanning_tree(mst_graph, algorithm="kruskal")

# Step 2: Generate hierarchical clustering tree
# Perform hierarchical clustering
linkage_matrix = linkage(squareform(distance_matrix), method="ward")

# Step 3: Plot MST
plt.figure(figsize=(6, 6))
pos = nx.spring_layout(mst, seed = 42)  # Position nodes using spring layout
nx.draw(
    mst, pos, with_labels=True, node_size=2500, font_size=16, edge_color="blue", node_color="lightgreen"
)
labels = nx.get_edge_attributes(mst, "weight")
nx.draw_networkx_edge_labels(mst, pos, edge_labels={k: f"{v:.1f}" for k, v in labels.items()})
plt.title("Minimum Spanning Tree (MST)", fontsize = 16)

plt.tight_layout()
plt.savefig('distance_02.jpg', dpi = 600)
plt.show()

# Step 4: Plot hierarchical clustering tree (dendrogram)
plt.figure(figsize=(6, 6))
dendrogram(linkage_matrix, labels=distance_df.columns, leaf_rotation=90, leaf_font_size=10)
#plt.title("Dendrogram of the Portfolio", fontsize = 12, weight = 'bold')
plt.xlabel("Stocks", fontsize = 16, weight = 'bold')
plt.ylabel("Distance", fontsize = 16, weight = 'bold')
plt.xticks(fontsize = 16, weight = 'bold')
plt.yticks(fontsize = 16, weight = 'bold')
plt.tight_layout()
plt.savefig('distance_03.jpg', dpi = 600)
plt.show()

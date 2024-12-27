import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming the CSV file is properly loaded and columns match as described
indices = ['IonQ', 'Rigetti Computing', 'Quantum Computing Inc.',
           'D-Wave Quantum', 'Alphabet', 'IBM', 
           'Microsoft', 'Nvidia', 'Defiance Quantum ETF', 'Global X Future Analytics Tech']

symbols = ['IONQ','RGTI','QUBT','QBTS','GOOGL','IBM','MSFT','NVDA','QTUM','AIQ']

df = pd.read_csv('quantum_technology_indices_prices.csv')
df = df.dropna(axis=0)
print(df)
print(df.columns)

fig, axes = plt.subplots(10, 10, figsize=(24, 20))
for i in range(len(indices)):
    for j in range(len(indices)):
        if i != j:
            #axes[j, i].axis('off')  # Turn off upper triangle
            ax = axes[i, j]
            ax.scatter(x=df[indices[i]], y=df[indices[j]], s=5, alpha=0.2, color='red')
            sns.regplot(
                x=df[indices[i]], 
                y=df[indices[j]], 
                ax=ax, 
                scatter=False, 
                color='red', 
                line_kws={'color': 'blue', 'linewidth': 2}
            )
            # Set xlabel and ylabel as symbolss of the assets
            ax.set_xlabel(symbols[i], fontsize = 16, weight = 'bold')
            ax.set_ylabel(symbols[j], fontsize = 16, weight = 'bold')

for i in range(len(indices)):
    for j in range(len(indices)):
        if i==j:
            axes[i,j].plot(df[indices[i]], color = 'blue', alpha = 0.5)
            axes[i,j].set_ylabel(symbols[i], fontsize = 15, weight = 'bold', color = 'blue')
            #ax.set_ylabel(None)
plt.tight_layout()
plt.savefig('0_correlation.jpg', dpi = 600)




# Add correlation heatmap
correlation_matrix = df[indices].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix, 
    annot=True, 
    cmap="coolwarm", 
    fmt=".2f", 
    linewidths=0.5, 
    cbar_kws={"shrink": 0.8}, 
    xticklabels=symbols, 
    yticklabels=symbols
)
# Increase font size for tick labels
plt.xticks(fontsize=14,weight = 'bold', rotation = 90)
plt.yticks(fontsize=14,weight = 'bold', rotation = 0)
plt.savefig('1_ correlation_heatmap.jpg', dpi  = 600)
plt.show()
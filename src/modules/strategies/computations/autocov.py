import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.modules.strategies.strategy_interface import StrategyInterface
from src.common.consts import CommonConsts

class PortfolioAutoCov(StrategyInterface):
    def __init__(self, price_matrix):
        self.price_matrix = price_matrix
        self.symbols = price_matrix.columns

    async def compute(self):
        max_lag = 30
        autocov_results = {symbol: [] for symbol in self.symbols}

        def autocovariance(series, lag):
            mean = series.mean()
            return np.mean((series[:-lag] - mean) * (series[lag:] - mean)) if lag > 0 else np.var(series)

        for symbol in self.symbols:
            series = self.price_matrix[symbol]
            for lag in range(1, max_lag + 1):
                autocov_results[symbol].append(autocovariance(series.values, lag))

        autocov_df = pd.DataFrame(autocov_results, index=range(1, max_lag + 1))
        plt.figure(figsize=(10, 8))
        sns.heatmap(autocov_df.T, annot=False, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title('Autocovariance Heatmap', fontsize=14, weight='bold')
        plt.tight_layout()
        plt.savefig(f'{CommonConsts.IMG_FOLDER}\\autocov_test.jpg', dpi=600)
        return plt
    
    def visualize(self):
        pass
    
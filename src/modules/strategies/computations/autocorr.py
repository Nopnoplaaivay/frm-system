import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.modules.strategies.strategy_interface import StrategyInterface
from src.common.consts import CommonConsts


class PortfolioAutoCorr(StrategyInterface):
    def __init__(self, price_matrix):
        self.price_matrix = price_matrix
        self.symbols = price_matrix.columns

    async def compute(self):
        max_lag = 30
        autocorr_results = {symbol: [] for symbol in self.symbols}
        
        for symbol in self.symbols:
            series = self.price_matrix[symbol]
            for lag in range(1, max_lag + 1):
                autocorr_results[symbol].append(series.autocorr(lag))

        autocorr_df = pd.DataFrame(autocorr_results, index=range(1, max_lag + 1))
        plt.figure(figsize=(12, 6))
        for symbol in self.symbols:
            plt.plot(autocorr_df.index, autocorr_df[symbol], label=symbol, marker='o', alpha=0.5)
        plt.title("Autocorrelation", fontsize=14, weight='bold')
        plt.xlabel("Lag", fontsize=12, weight='bold')
        plt.ylabel("Autocorrelation", fontsize=12, weight='bold')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{CommonConsts.IMG_FOLDER}\\autocorr_test.jpg', dpi=600)

        return plt

    def visualize(self):
        pass



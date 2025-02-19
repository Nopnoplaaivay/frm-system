import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from src.common.consts import CommonConsts
from src.services.strategies.strategy_interface import StrategyInterface

class PortfolioAutoCorr(StrategyInterface):
    def __init__(self, price_matrix):
        self.price_matrix = price_matrix
        self.symbols = price_matrix.columns


    def analyze(self, df: pd.DataFrame):
        max_lag = 30
        autocorr_results = {symbol: [] for symbol in self.symbols}
        autocov_results = {symbol: [] for symbol in self.symbols}

        for symbol in self.symbols:
            series = df[symbol]
            for lag in range(1, max_lag + 1):
                autocorr_results[symbol].append(series.autocorr(lag))
                autocov_results[symbol].append(self._autocovariance(series.values, lag))

        return pd.DataFrame(autocorr_results, index=range(1, max_lag + 1)), \
               pd.DataFrame(autocov_results, index=range(1, max_lag + 1))

    def visualize(self, df):
        indices = CommonConsts.ticker_model
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        autocorr_df, autocov_df = self.analyze(df)

        for symbol in self.symbols:
            axs[0, 0].plot(autocorr_df.index, autocorr_df[symbol], label=f'{symbol}', marker = 'o', alpha = 0.5, lw = 1)

        axs[0, 0].set_ylabel('Autocorrelation', fontsize = 12, weight = 'bold')
        axs[0, 0].set_xlabel('Lag', fontsize = 12, weight = 'bold')
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        for symbol in self.symbols:
            axs[0, 1].plot(autocov_df.index, autocov_df[symbol], label=f'{symbol}', marker = 'o', alpha = 0.5, lw = 1)

        axs[0, 1].set_ylabel('Autocovariance', fontsize = 12, weight = 'bold')
        axs[0, 1].set_xlabel('Lag', fontsize = 12, weight = 'bold')
        axs[0, 1].grid(True)

        sns.heatmap(autocorr_df.T, cmap="coolwarm", annot=False, ax=axs[1, 0])
        axs[1, 0].set_title('Autocorrelation Heatmap', fontsize = 12, weight = 'bold')
        axs[1, 0].set_xlabel('Lag', fontsize = 12, weight = 'bold')
        axs[1, 0].set_ylabel('Symbols', fontsize = 12, weight = 'bold')

        sns.heatmap(autocov_df.T, cmap="coolwarm", annot=False, ax=axs[1, 1])
        axs[1, 1].set_title('Autocovariance Heatmap', fontsize = 12, weight = 'bold')
        axs[1, 1].set_xlabel('Lag', fontsize = 12, weight = 'bold')
        axs[1, 1].set_ylabel('Symbols', fontsize = 12, weight = 'bold')

        plt.tight_layout()
        plt.savefig(f'{CommonConsts.IMG_FOLDER}\\autocorr.jpg', dpi = 600)

    def autocorrelation(self, series, lag):
        return series.autocorr(lag)
    
    def autocovariance(self, series, lag):
        mean = series.mean()
        return np.mean((series[:-lag] - mean) * (series[lag:] - mean)) if lag > 0 else np.var(series)

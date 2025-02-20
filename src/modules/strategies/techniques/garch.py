import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from arch import arch_model

from src.modules.strategies.strategy_interface import StrategyInterface
from src.common.consts import CommonConsts

class PortfolioGarch(StrategyInterface):
    def __init__(self, price_matrix):
        self.price_matrix = price_matrix
        self.symbols = price_matrix.columns

    def compute(self):
        return self.price_matrix[self.symbols].pct_change().dropna()

    def render_chart(self):
        # Compute returns
        returns = self.compute()

        """GARCH Model for Volatility"""
        st.subheader("GARCH Model for Volatility")
        vol_model = 'Garch'
        num_symbols = len(self.symbols)
        fig, axes = plt.subplots(nrows=(num_symbols + 1) // 2, ncols=2, figsize=(12, 10))
        axes = axes.flatten()

        results = {}
        for i, symbol in enumerate(self.symbols):
            model = arch_model(returns[symbol]*1000, vol=vol_model, p=1, q=1, 
                mean='Constant', dist='Normal')
            res = model.fit(disp='off')
            results[symbol] = res

            res.conditional_volatility.plot(ax=axes[i], color = 'red')
            axes[i].set_ylabel(r'$\sigma \times 10^3$', fontsize = 14, weight = 'bold')
            axes[i].set_title(label=f'{symbol}', fontsize=16, weight = 'bold')
            axes[i].grid(True)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        # plt.savefig(f'{CommonConsts.IMG_FOLDER}\\Garch.jpg', dpi = 600)
        st.pyplot(fig)

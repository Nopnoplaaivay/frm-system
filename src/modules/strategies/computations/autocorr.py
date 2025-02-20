import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import altair as alt

from src.modules.strategies.strategy_interface import StrategyInterface
from src.common.consts import CommonConsts


class PortfolioAutoCorr(StrategyInterface):
    def __init__(self, price_matrix):
        self.price_matrix = price_matrix
        self.symbols = price_matrix.columns

    def compute(self):
        max_lag = 30
        autocorr_results = {symbol: [] for symbol in self.symbols}
        
        for symbol in self.symbols:
            series = self.price_matrix[symbol]
            for lag in range(1, max_lag + 1):
                autocorr_results[symbol].append(series.autocorr(lag))

        autocorr_df = pd.DataFrame(autocorr_results, index=range(1, max_lag + 1))
        return autocorr_df

    def render_plot(self):
        # Compute autocorrelation data
        autocorr_df = self.compute()

        melted_df = autocorr_df.reset_index().melt(id_vars='index', var_name='symbol', value_name='autocorrelation')
        melted_df = melted_df.rename(columns={'index': 'Lag', 'symbol': 'Symbol', 'autocorrelation': 'Autocorrelation'})
        y_min = melted_df['Autocorrelation'].min()

        chart = alt.Chart(melted_df).mark_line().encode(
            x='Lag:Q',
            y=alt.Y('Autocorrelation:Q', scale=alt.Scale(domain=[y_min, 1])),
            color='Symbol:N'
        ).properties(
            width=600,
            height=400
        )

        # Display the chart using st.altair_chart
        st.altair_chart(chart, use_container_width=True)

        # plt.figure(figsize=(12, 6))
        # for symbol in self.symbols:
        #     plt.plot(autocorr_df.index, autocorr_df[symbol], label=symbol, marker='o', alpha=0.5)
        # plt.title("Autocorrelation", fontsize=14, weight='bold')
        # plt.xlabel("Lag", fontsize=12, weight='bold')
        # plt.ylabel("Autocorrelation", fontsize=12, weight='bold')
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(f'{CommonConsts.IMG_FOLDER}\\autocorr_test.jpg', dpi=600)
        # return plt

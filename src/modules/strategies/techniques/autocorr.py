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

    def render_chart(self):
        autocorr_df = self.compute()

        melted_df = autocorr_df.reset_index().melt(id_vars='index', var_name='symbol', value_name='autocorrelation')
        melted_df.rename(columns={"index": "lag"}, inplace=True)
        y_min = melted_df['autocorrelation'].min()


        """Correlation Line Chart"""
        line_chart = alt.Chart(melted_df).mark_line().encode(
            x=alt.X('lag:Q', title='Lag'),
            y=alt.Y('autocorrelation:Q', scale=alt.Scale(domain=[y_min, 1]), title='Autocorrelation'),
            color='symbol:N'
        ).properties(
            width=600,
            height=400
        )
        st.subheader('Autocorrelation Line Chart')
        st.altair_chart(line_chart, use_container_width=True)


        """Correlation Heatmap"""
        heatmap = (
            alt.Chart(melted_df)
            .mark_rect()
            .encode(
                x=alt.X("lag:O", title="Lag", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("symbol:N", title="Symbol"),
                color=alt.Color(
                    "autocorrelation:Q",
                    scale=alt.Scale(scheme="blueorange"),  # Use a diverging color scheme
                    legend=alt.Legend(title="Autocorrelation"),
                ),
                tooltip=["lag", "symbol", "autocorrelation"],
            )
            .properties(
                width=600,
                height=400
            )
        )
        st.subheader("Autocorrelation Heatmap")
        st.altair_chart(heatmap, use_container_width=True)


import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

from src.modules.strategies.strategy_interface import StrategyInterface
from src.common.consts import CommonConsts

class PortfolioAutoCov(StrategyInterface):
    def __init__(self, price_matrix):
        self.price_matrix = price_matrix
        self.symbols = price_matrix.columns

    def compute(self):
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
        return autocov_df

    def render_chart(self):
        autocov_df = self.compute()

        melted_df = autocov_df.reset_index().melt(id_vars="index", var_name="symbol", value_name="autocovariance")
        melted_df.rename(columns={"index": "lag"}, inplace=True)
        y_min = melted_df['autocovariance'].min()
        y_max = melted_df['autocovariance'].max()


        """Correlation Line Chart"""
        line_chart = alt.Chart(melted_df).mark_line().encode(
            x=alt.X('lag:Q', title='Lag'),
            y=alt.Y('autocovariance:Q', scale=alt.Scale(domain=[y_min, y_max]), title='Autocovariance'),
            color='symbol:N'
        ).properties(
            width=600,
            height=400
        )
        st.subheader('Autocovariance Line Chart')
        st.altair_chart(line_chart, use_container_width=True)

        heatmap = (
            alt.Chart(melted_df)
            .mark_rect()
            .encode(
                x=alt.X("lag:O", title="Lag", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("symbol:N", title="Symbol"),
                color=alt.Color(
                    "autocovariance:Q",
                    scale=alt.Scale(scheme="blueorange"),  # Use a diverging color scheme
                    legend=alt.Legend(title="Autocovariance"),
                ),
                tooltip=["lag", "symbol", "autocovariance"],
            )
            .properties(
                width=600,
                height=400
            )
        )
        st.subheader("Autocovariance Heatmap")
        st.altair_chart(heatmap, use_container_width=True)
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt

from src.modules.strategies.strategy_interface import StrategyInterface
from src.common.consts import CommonConsts

class PortfolioEDA(StrategyInterface):
    def __init__(self, price_matrix):
        self.price_matrix = price_matrix
        self.symbols = price_matrix.columns

    def compute(self):
        pass

    def render_chart(self):
        """Scatterplot Matrix"""
        st.subheader("Scatterplot Matrix")
        stocks_num = len(self.symbols)
        fig, axes = plt.subplots(stocks_num, stocks_num, figsize=(24, 20))
        for i in range(len(self.symbols)):
            for j in range(len(self.symbols)):
                if i != j:
                    ax = axes[i, j]
                    ax.scatter(x=self.price_matrix[self.symbols[i]], y=self.price_matrix[self.symbols[j]], s=5, alpha=0.2, color='red')
                    sns.regplot(
                        x=self.price_matrix[self.symbols[i]],
                        y=self.price_matrix[self.symbols[j]],
                        ax=ax,
                        scatter=False,
                        color='red',
                        line_kws={'color': 'blue', 'linewidth': 2}
                    )
                    ax.set_xlabel(self.symbols[i], fontsize=16, weight='bold')
                    ax.set_ylabel(self.symbols[j], fontsize=16, weight='bold')
                else:
                    axes[i, j].plot(self.price_matrix[self.symbols[i]], color='blue', alpha=0.5)
                    axes[i, j].set_ylabel(self.symbols[i], fontsize=15, weight='bold', color='blue')
        st.pyplot(fig)

        """Correlation Heatmap"""
        st.subheader("Correlation Heatmap")
        correlation_matrix = self.price_matrix[self.symbols].corr()
        melted_corr = correlation_matrix.reset_index().melt(id_vars="index", var_name="variable", value_name="correlation")
        melted_corr.rename(columns={"index": "symbol"}, inplace=True)

        heatmap = (
            alt.Chart(melted_corr)
            .mark_rect()
            .encode(
                x=alt.X("symbol:N", title="Symbol"),
                y=alt.Y("variable:N", title="Variable"),
                color=alt.Color("correlation:Q", scale=alt.Scale(scheme="blueorange"), legend=alt.Legend(title="Correlation")),
                tooltip=["symbol", "variable", "correlation"],
            )
            .properties(width=600, height=400)
        )

        text = heatmap.mark_text(baseline="middle").encode(
            text=alt.Text("correlation:Q", format=".2f"),
            color=alt.condition(
                "datum.correlation > 0",
                alt.value("white"),  # White text for positive values
                alt.value("black"),  # Black text for negative values
            ),
        )
        st.altair_chart(heatmap + text, use_container_width=True)
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage

from src.modules.strategies.strategy_interface import StrategyInterface
from src.common.consts import CommonConsts

class PortfolioDistance(StrategyInterface):
    def __init__(self, price_matrix):
        self.price_matrix = price_matrix
        self.symbols = price_matrix.columns

    def compute(self):
        lag = 5
        log_DF = pd.DataFrame([])

        # Compute log returns and standardize them
        for index in range(len(self.symbols)):
            symbol = self.price_matrix.columns[index]
            prices = self.price_matrix[symbol]
            log_returns = np.log(prices / prices.shift(lag)).dropna()
            log_returns_mean = log_returns.mean()
            log_returns_std = log_returns.std()
            log_returns = (log_returns - log_returns_mean) / (log_returns_std)
            log_DF[symbol] = log_returns

        standardized_df = (log_DF - log_DF.mean()) / log_DF.std()

        # Compute distance matrix
        distance_matrix = squareform(pdist(standardized_df.T, metric="euclidean"))
        distance_df = pd.DataFrame(distance_matrix, index=self.symbols, columns=self.symbols)

        # Build Minimum Spanning Tree (MST)
        mst_graph = nx.Graph()
        for i, stock1 in enumerate(distance_df.columns):
            for j, stock2 in enumerate(distance_df.columns):
                if i < j:
                    mst_graph.add_edge(stock1, stock2, weight=distance_df.iloc[i, j])
        mst = nx.minimum_spanning_tree(mst_graph, algorithm="kruskal")

        # Generate hierarchical clustering tree
        linkage_matrix = linkage(squareform(distance_matrix), method="ward")

        return distance_df, mst, linkage_matrix

    def render_chart(self):
        # Compute data
        distance_df, mst, linkage_matrix = self.compute()

        """Plot pairwise distance matrix (heatmap)"""
        melted_df = distance_df.reset_index().melt(id_vars="index", var_name="symbol", value_name="distance")
        melted_df.rename(columns={"index": "stock"}, inplace=True)

        heatmap = (
            alt.Chart(melted_df)
            .mark_rect()
            .encode(
                x=alt.X("stock:N", title="Symbol"),
                y=alt.Y("symbol:N", title="Symbol"),
                color=alt.Color("distance:Q", scale=alt.Scale(scheme="blueorange")),
                tooltip=["stock", "symbol", "distance"],
            )
            .properties(width=600, height=400)
        )

        text = heatmap.mark_text(baseline="middle").encode(
            text=alt.Text("distance:Q", format=".1f"),
            color=alt.condition(
                "datum.distance > 0",
                alt.value("white"),  # White text for positive values
                alt.value("black"),  # Black text for negative values
            ),
        )

        st.subheader("Pairwise Distance Heatmap")
        st.altair_chart(heatmap + text, use_container_width=True)

        # Plot Minimum Spanning Tree (MST)
        pos = nx.spring_layout(mst, seed=42)
        edge_x = []
        edge_y = []
        for edge in mst.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        node_x = [pos[node][0] for node in mst.nodes()]
        node_y = [pos[node][1] for node in mst.nodes()]

        edge_trace = {
            "type": "scatter",
            "x": edge_x,
            "y": edge_y,
            "mode": "lines",
            "line": {"color": "blue", "width": 1},
        }

        node_trace = {
            "type": "scatter",
            "x": node_x,
            "y": node_y,
            "mode": "markers+text",
            "marker": {"size": 15, "color": "lightgreen"},
            "text": list(mst.nodes()),
            "textposition": "top center",
            "textfont": {"size": 12, "color": "black"},
        }

        fig = {
            "data": [edge_trace, node_trace],
            "layout": {
                "showlegend": False,
                "hovermode": "closest",
                "margin": {"b": 20, "l": 5, "r": 5, "t": 40},
                "xaxis": {"showgrid": False, "zeroline": False, "showticklabels": False},
                "yaxis": {"showgrid": False, "zeroline": False, "showticklabels": False},
            },
        }
        st.subheader("Minimum Spanning Tree")
        st.plotly_chart(fig, use_container_width=True)

        plt.figure(figsize=(6, 6))
        dendrogram(linkage_matrix, labels=distance_df.columns, leaf_rotation=90, leaf_font_size=10)
        plt.xlabel("Stocks", fontsize = 12)
        plt.ylabel("Distance", fontsize = 12)
        plt.xticks(fontsize = 10)
        plt.yticks(fontsize = 12)
        plt.tight_layout()

        st.subheader("Hierarchical Clustering Tree")
        st.pyplot(plt)
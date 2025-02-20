import asyncio
import pandas as pd
import streamlit as st
from src.modules.yfinance import PortfolioMatrix
from src.modules.pipelines import PortfolioPipeline
from src.common.consts import CommonConsts, YfinanceConsts


async def build_price_matrix(selected_symbols, time_range):
    # Fetch the portfolio price matrix
    portfolio_price_matrix = await PortfolioMatrix.build(symbols=selected_symbols, time_range=time_range)
    return portfolio_price_matrix


def streamlit_app():
    st.title("Financial Risk Management App")

    """Side bar"""
    with st.sidebar:
        """Build portfolio"""
        selected_symbols = st.multiselect("Build Portfolio", CommonConsts.STOCKS_LIST)
        if not selected_symbols:
            st.warning("Please select at least one stock symbol.")
            return
        
        """Select time range"""
        time_range = st.radio("Select time range", YfinanceConsts.AVAILABLE_RANGES)
        time_range = time_range.lower()
        if not time_range:
            st.warning("Please select a time range.")
            return
        
        """Build portfolio price matrix"""
        if st.button("Build"):
            with st.spinner("Loading matrix..."):
                try:
                    portfolio_price_matrix = asyncio.run(build_price_matrix(selected_symbols, time_range))

                    st.session_state.portfolio_price_matrix = portfolio_price_matrix
                    st.success("Matrix loaded successfully!")
                except Exception as e:
                    st.error(f"An error occurred while loading the matrix: {e}")
                    raise e

    """Show portfolio price matrix"""
    with st.expander("View Portfolio Price Matrix", expanded=False):
        if "portfolio_price_matrix" in st.session_state:

            # Sort index to display the most recent date first
            sorted_matrix = st.session_state.portfolio_price_matrix.sort_index(ascending=False)
            st.dataframe(sorted_matrix)

    """Analyze the portfolio"""
    if "portfolio_price_matrix" in st.session_state:
        if st.button("Analyze"):
            with st.spinner("Computing..."):
                try:
                    PortfolioPipeline.run(st.session_state.portfolio_price_matrix)
                except Exception as e:
                    st.error(f"An error occurred while running the pipeline: {e}")
                    raise e

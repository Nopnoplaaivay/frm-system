import streamlit as st
from src.modules.strategies.computations import (
    PortfolioAutoCov,
    PortfolioAutoCorr,
    PortfolioDistance
)
from src.modules.yfinance import PortfolioMatrix
from src.modules.pipelines import PortfolioPipeline
from src.common.consts import CommonConsts


async def build_price_matrix(selected_symbols):
    # Fetch the portfolio price matrix
    portfolio_price_matrix = await PortfolioMatrix.build(symbols=selected_symbols)
    return portfolio_price_matrix


# Asynchronous function to run the pipeline
def run_pipeline(portfolio_price_matrix):
    # Run pipeline (Get plots)
    pipeline = PortfolioPipeline(price_matrix=portfolio_price_matrix)
    plots = pipeline.run()
    return plots


def streamlit_app():
    st.title("Financial Risk Management App")

    """Select stocks"""
    selected_symbols = st.multiselect("Build Portfolio", CommonConsts.model_symbols)
    if not selected_symbols:
        st.warning("Please select at least one stock symbol.")
        return

    if st.button("Build"):
        with st.spinner("Loading matrix..."):
            try:
                # Load price matrix
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                portfolio_price_matrix = loop.run_until_complete(build_price_matrix(selected_symbols))
                loop.close()

                # Store the matrix in session state
                st.session_state.portfolio_price_matrix = portfolio_price_matrix
                st.success("Matrix loaded successfully!")
            except Exception as e:
                st.error(f"An error occurred while loading the matrix: {e}")
                return

    # Display the DataFrame if it exists in session state
    if "portfolio_price_matrix" in st.session_state:
        st.write("Portfolio Price Matrix:")
        st.dataframe(st.session_state.portfolio_price_matrix)

    # Step 3: Run Pipeline Button
    if "portfolio_price_matrix" in st.session_state:
        if st.button("Visualize"):
            with st.spinner("Computing..."):
                try:
                    st.success("Visualizing executed successfully!")

                    PortfolioAutoCorr(price_matrix=st.session_state.portfolio_price_matrix).render_plot()

                    # plots = run_pipeline(st.session_state.portfolio_price_matrix)
                    # for (plot_name, result) in plots:
                    #     st.subheader(plot_name)
                    #     st.pyplot(result)
                except Exception as e:
                    st.error(f"An error occurred while running the pipeline: {e}")
                    raise e

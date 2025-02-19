import streamlit as st
from src.modules.strategies.computations import (
    PortfolioAutoCov,
    PortfolioAutoCorr,
    PortfolioDistance
)
from src.modules.yfinance import PortfolioMatrix
from src.modules.pipelines import PortfolioPipeline
from src.common.consts import CommonConsts

# Asynchronous function to prepare data
async def prepare_data(selected_symbols):
    # Fetch the portfolio price matrix
    portfolio_price_matrix = await PortfolioMatrix.build(symbols=selected_symbols)
    return portfolio_price_matrix

# Asynchronous function to run the pipeline
async def run_pipeline(portfolio_price_matrix):
    # Run pipeline (Get plots)
    pipeline = PortfolioPipeline(price_matrix=portfolio_price_matrix)
    results = await pipeline.run()
    return results

# Streamlit app
def streamlit_app():
    st.title("Financial Risk Management App")

    # Step 1: Select stock symbols
    selected_symbols = st.multiselect("Select Stock Symbols", CommonConsts.model_symbols)
    if not selected_symbols:
        st.warning("Please select at least one stock symbol.")
        return

    # Step 2: Load Matrix Button
    if st.button("Load Matrix"):
        with st.spinner("Loading matrix..."):
            try:
                # Use Streamlit's asyncio integration
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                portfolio_price_matrix = loop.run_until_complete(prepare_data(selected_symbols))
                loop.close()

                # Store the matrix in session state
                st.session_state.portfolio_price_matrix = portfolio_price_matrix
                st.success("Matrix loaded successfully!")
                st.write("Portfolio Price Matrix:")
                st.dataframe(portfolio_price_matrix)  # Display the matrix as a dataframe
            except Exception as e:
                st.error(f"An error occurred while loading the matrix: {e}")
                return

    # Step 3: Run Pipeline Button
    if "portfolio_price_matrix" in st.session_state:
        if st.button("Run Pipeline"):
            with st.spinner("Running pipeline..."):
                try:
                    # Use Streamlit's asyncio integration
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    results = loop.run_until_complete(run_pipeline(st.session_state.portfolio_price_matrix))
                    loop.close()

                    st.success("Pipeline executed successfully!")

                    # Display results
                    for i, result in enumerate(results):
                        st.subheader(f"Plot {i + 1}")
                        st.pyplot(result)
                except Exception as e:
                    st.error(f"An error occurred while running the pipeline: {e}")

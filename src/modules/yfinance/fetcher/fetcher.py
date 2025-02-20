import aiohttp
import asyncio
import datetime
import pandas as pd
import streamlit as st

from src.utils.logger import LOGGER
from src.common.consts import YfinanceConsts


class YfinanceFetcher:
    MARKET = "VN"  # Assuming this is a constant for the market suffix

    @classmethod
    async def call_hist_price_api(cls, symbol, interval, time_range) -> pd.DataFrame:
        if time_range not in YfinanceConsts.VALID_RANGES:
            raise ValueError(f"Invalid range value: {time_range}")
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}.{cls.MARKET}?interval={interval}&range={time_range}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    data = await response.json()

                    # Extract relevant data from the JSON response
                    quote_data = data["chart"]["result"][0]["indicators"]["quote"][0]
                    df = pd.DataFrame(quote_data)

                    # Add timestamp column
                    df["timestamp"] = data["chart"]["result"][0]["timestamp"]
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
                    df["timestamp"] = df["timestamp"].dt.date

                    # Check if stock has enough records range (today - range)
                    if cls.check_data_sufficiency(symbol, df, time_range) == False:
                        return None

                    df.set_index("timestamp", inplace=True)
                    df.columns = [
                        f"{symbol}_{col}" for col in df.columns if col != "timestamp"
                    ]
                    return df
        except Exception as e:
            LOGGER.error(f"Failed to fetch data for {symbol}: {e}")
            st.warning(f"Yfinance doesn't provide data for {symbol}")
            return pd.DataFrame()

    @classmethod
    async def download(
        cls, symbols: list = ["BID"], interval: str = "1d", time_range: str = "5y"
    ) -> pd.DataFrame:
        tasks = [
            cls.call_hist_price_api(
                symbol=symbol, interval=interval, time_range=time_range
            )
            for symbol in symbols
        ]
        results = await asyncio.gather(*tasks)
        return pd.concat(results, axis=1)

    @staticmethod
    def check_data_sufficiency(symbol: str, df: pd.DataFrame, time_range: str):
        # Parse the requested range into years or months
        end_date = datetime.datetime.now().date()
        if time_range.endswith("y"):  # Years
            years = int(time_range[:-1])
            expected_start_month = (end_date.month - (years * 12)) % 12 or 12
            expected_start_year = end_date.year - (years + ((end_date.month - 1) // 12))
        elif time_range.endswith("mo"):  # Months
            months = int(time_range[:-2])
            expected_start_month = (end_date.month - months) % 12 or 12
            expected_start_year = end_date.year - ((end_date.month - months - 1) // 12)
        else:
            raise ValueError(f"Unsupported range format: {time_range}")

        actual_start_date = df["timestamp"].min()

        if actual_start_date.year > expected_start_year or (
            actual_start_date.year == expected_start_year
            and actual_start_date.month > expected_start_month
        ):
            LOGGER.warning(
                f"Insufficient data for {symbol}. "
                f"Expected start month/year: {expected_start_month}/{expected_start_year}, "
                f"but got: {actual_start_date.month}/{actual_start_date.year}. "
                f"Removing {symbol} from the portfolio."
            )

            st.warning(
                f"Insufficient data for {symbol}. "
                f"Expected data range from: {expected_start_month}/{expected_start_year} - {end_date.month}/{end_date.year}\n"
                f"Yfinance only provides from: {actual_start_date.month}/{actual_start_date.year}. - {end_date.month}/{end_date.year}. \n"
                f"Removing {symbol} from the portfolio."
            )

            return False
        return True

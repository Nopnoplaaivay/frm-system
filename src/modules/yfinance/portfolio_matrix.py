import pandas as pd

from src.modules.yfinance.fetcher import YfinanceFetcher
from src.modules.yfinance.processor import YfinanceProcessor


class PortfolioMatrix:
    @classmethod
    async def build(cls, symbols: list, interval: str = "1d", time_range: str = "5y") -> pd.DataFrame:
        raw_data = await YfinanceFetcher.download(symbols=symbols, interval=interval, time_range=time_range)
        return YfinanceProcessor.process_price_data(df=raw_data)
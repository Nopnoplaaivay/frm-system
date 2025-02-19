import asyncio
import time

from src.modules.yfinance import PortfolioMatrix
from src.modules.pipelines import PortfolioPipeline


async def test():
    start_time = time.time()
    symbols = ['BID', 'CTG', 'VCB', 'VPB', 'EIB', 'HDB', 'MBB', 'STB', 'ACB', 'TCB']
    interval = '1d'
    time_range = '5y'

    matrix = await PortfolioMatrix.build(symbols=symbols, interval=interval, time_range=time_range)
    pipeline = PortfolioPipeline(price_matrix=matrix)
    results = await pipeline.run()

if __name__ == '__main__':
    asyncio.run(test())
import asyncio
import time

from src.modules.yfinance import PortfolioMatrix


async def test():
    start_time = time.time()
    symbols = ['BID', 'CTG', 'VCB', 'VPB', 'EIB', 'HDB', 'MBB', 'STB', 'ACB', 'TCB']
    interval = '1d'
    time_range = '5y'

    matrix = await PortfolioMatrix.build(symbols=symbols, interval=interval, time_range=time_range)
    print(matrix)
    print(f"Build matrix time: {time.time() - start_time}")

if __name__ == '__main__':
    asyncio.run(test())
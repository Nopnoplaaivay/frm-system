import asyncio
import time

from src.modules.yfinance.data import YfinanceData


async def test():
    start_time = time.time()
    symbols = ['BID', 'CTG', 'VCB', 'VPB', 'EIB', 'HDB', 'MBB', 'STB', 'ACB', 'TCB']
    raw_data = await YfinanceData.download(symbols=symbols)
    print(f"Download data time: {time.time() - start_time}")
    raw_data.to_csv('raw_data.csv')
    print(f"Saved data time: {time.time() - start_time}")
    # cleaned_data = DataCleaner.process_price_data(raw_data)
    # print(f"Clean data time: {time.time() - start_time}")
    # cleaned_data.to_csv('processed_data.csv')

if __name__ == '__main__':
    asyncio.run(test())
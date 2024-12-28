import pandas as pd

from src.common.consts import CommonConsts
from src.services.strategies import EDAPortfolio
from src.services.statistical_analysis_service import StatisticalAnalysisService
from src.data.processors import Processors

print(CommonConsts.ROOT_FOLDER)
print(CommonConsts.IMG_FOLDER)

data_path = 'src/data/fiinx/raw_data.csv'
raw_data = pd.read_csv(data_path)
df = Processors.transform(raw_data)
print(df)

strategy = EDAPortfolio()
service = StatisticalAnalysisService(strategy)
service.visualize(df)

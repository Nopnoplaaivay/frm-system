import pandas as pd

from src.common.consts import CommonConsts
from src.data.processors import Processors
from src.services.statistical_analysis_service import StatisticalAnalysisService
from src.services.strategies import (
    PortfolioEDA,
    PortfolioAutoCorr,
    PortfolioSpectralDensity,
    PortfolioDistance,
    PortfolioGarch,
    PortfolioRatios,
)

# print(CommonConsts.ROOT_FOLDER)
# print(CommonConsts.IMG_FOLDER)

data_path = "src/data/fiinx/raw_data.csv"
raw_data = pd.read_csv(data_path)
df = Processors.transform(raw_data)

# strategy = PortfolioEDA()
# strategy = PortfolioAutoCorr()
# strategy = PortfolioSpectralDensity()
# strategy = PortfolioDistance()
# strategy = PortfolioGarch()
strategy = PortfolioRatios()
service = StatisticalAnalysisService(strategy)
service.visualize(df)

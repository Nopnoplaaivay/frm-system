import asyncio

from src.modules.strategies.techniques import (
    PortfolioAutoCorr,
    PortfolioAutoCov,
    PortfolioDistance,
    PortfolioEDA,
    PortfolioGarch,
    PortfolioPerformance,
    PortfolioSpectralDensity,
    PortfolioStationary
)
from src.utils.logger import LOGGER


class PortfolioPipeline:
    @classmethod
    def run(cls, price_matrix):
        computations = [
            PortfolioAutoCorr(price_matrix),
            PortfolioAutoCov(price_matrix),
            PortfolioDistance(price_matrix),
            PortfolioEDA(price_matrix),
            PortfolioGarch(price_matrix),
            PortfolioPerformance(price_matrix),
            PortfolioSpectralDensity(price_matrix),
            PortfolioStationary(price_matrix)
        ]
        for job in computations:
            job.render_chart()

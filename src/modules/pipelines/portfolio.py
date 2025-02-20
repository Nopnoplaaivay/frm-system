import asyncio

from src.modules.strategies.computations import (
    PortfolioAutoCorr,
    PortfolioAutoCov,
    PortfolioDistance,
)
from src.utils.logger import LOGGER


class PortfolioPipeline:
    def __init__(self, price_matrix):
        self.price_matrix = price_matrix

    def run(self):
        computations = [
            PortfolioAutoCorr(price_matrix=self.price_matrix),
            # PortfolioAutoCov(price_matrix=self.price_matrix),
            # PortfolioDistance(price_matrix=self.price_matrix),
        ]

        plots = []
        for task in computations:
            LOGGER.info(f"Computing {task.__class__.__name__}...")
            plots.append((task.__class__.__name__, task.compute()))

        return plots

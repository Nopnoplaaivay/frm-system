import asyncio

from src.modules.strategies.computations import (
    PortfolioAutoCorr,
    PortfolioAutoCov,
    PortfolioDistance,
)


class PortfolioPipeline:
    def __init__(self, price_matrix):
        self.price_matrix = price_matrix

    async def run(self):
        tasks = [
            PortfolioAutoCorr(price_matrix=self.price_matrix).compute(),
            PortfolioAutoCov(price_matrix=self.price_matrix).compute(),
            # PortfolioDistance(price_matrix=self.price_matrix).compute(),
        ]

        results = await asyncio.gather(*tasks)
        return results

from abc import ABC, abstractmethod
import pandas as pd

class StrategyInterface(ABC):
    @abstractmethod
    async def compute(self):
        pass

    @abstractmethod
    def visualize(self, df: pd.DataFrame):
        pass
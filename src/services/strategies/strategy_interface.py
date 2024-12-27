import pandas as pd

from abc import ABC, abstractmethod

class StrategyInterface(ABC):

    @abstractmethod
    def analyze(self, df: pd.DataFrame):
        pass
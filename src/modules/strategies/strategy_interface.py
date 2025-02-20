from abc import ABC, abstractmethod
import pandas as pd

class StrategyInterface(ABC):
    @abstractmethod
    def compute(self):
        pass
class StatisticalAnalysisService:
    def __init__(self, strategy):
        self._strategy = strategy

    def set_strategy(self, strategy):
        self._strategy = strategy

    def analyze(self, data):
        return self._strategy.analyze(data)

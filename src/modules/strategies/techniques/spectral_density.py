import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.signal import welch

from src.modules.strategies.strategy_interface import StrategyInterface

class PortfolioSpectralDensity(StrategyInterface):
    def __init__(self, price_matrix):
        self.price_matrix = price_matrix
        self.symbols = price_matrix.columns

    def compute(self):
        log_returns = np.log(self.price_matrix[self.symbols] / self.price_matrix[self.symbols].shift(1)).dropna()
        return log_returns
    

    def render_chart(self):
        log_returns = self.compute()
        
        """Portfolio Spectral Density"""
        st.subheader('Portfolio Spectral Density')
        plt.figure(figsize=(10, 4))
        for symbol in self.symbols:
            freqs, psd = welch(log_returns[symbol], fs=1.0, nperseg=min(256, len(log_returns[symbol])))
            plt.plot(freqs, psd, label=symbol)

        plt.xlabel('Frequency', fontsize = 12)
        plt.ylabel('Power Spectral Density', fontsize = 12)
        plt.yscale('log')  # Log scale for better visualization of spectral features
        plt.legend(ncol = 5)
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(plt)
        # plt.savefig(f'{CommonConsts.IMG_FOLDER}\\spectral_density.jpg', dpi = 600)

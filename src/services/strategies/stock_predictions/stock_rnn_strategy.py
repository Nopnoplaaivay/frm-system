import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from src.common.consts import CommonConsts
from src.data.processors import Processors
from src.services.strategies.stock_predictions.models import StockRNN
from src.services.strategies.stock_predictions.trainers import ModelTrainer
from src.services.strategies.strategy_interface import StrategyInterface
from src.utils.logger import LOGGER


class StockRNNStratgy(StrategyInterface):
    def analyze(self, df: pd.DataFrame):
        return df
    
    def visualize(self, df: pd.DataFrame):
        indices = CommonConsts.ticker_model
        df = df.reset_index(drop=True)

        for symbol in indices:
            symbol_df = df[symbol]
            data_dict = Processors.prepare_data(symbol_df=symbol_df)

            # Initialize model & trainer
            model = StockRNN(
                input_size=1,
                hidden_size=CommonConsts.HIDDEN_SIZE,
                num_layers=CommonConsts.NUM_LAYERS,
                output_size=1
            )

            trainer = ModelTrainer(
                model=model,
                criterion=nn.MSELoss(),
                optimizer=torch.optim.Adam(model.parameters(), lr=CommonConsts.LEARNING_RATE)
            )

            # Train model
            LOGGER.info(f"\nTraining model for {symbol}")
            trainer.train_model(train_loader=data_dict['train_loader'])

            # Evaluate model
            predictions = trainer.predict(X_test=data_dict['X_test'])
            y_test_np = data_dict['y_test'].numpy()

            # Inverse transform predictions
            scaler = data_dict['scaler']
            predictions = scaler.inverse_transform(predictions)
            y_test_np = scaler.inverse_transform(y_test_np)

            # Future predictions
            last_sequence = data_dict['X_test'][-2].reshape(1, CommonConsts.SEQUENCE_LENGTH, 1)
            future_predictions = trainer.predict_future_price(last_sequence)
            future_predictions = scaler.inverse_transform(future_predictions)

            # Visualize results
            self.plot_prediction(
                y_test=y_test_np,
                predictions=predictions,
                future_predictions=future_predictions,
                symbol=symbol
            )

    def plot_prediction(self,         
        y_test: np.ndarray,
        predictions: np.ndarray,
        future_predictions: np.ndarray,
        symbol: str
    ):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        
        # Test set predictions
        axes[0].plot(y_test, label='Actual Prices', color='blue')
        axes[0].plot(predictions, label='Predicted Prices', marker='o', alpha=0.5, color='red')
        axes[0].legend()
        axes[0].set_title(f'{symbol} Price Prediction (Test Set)', weight='bold')
        axes[0].set_xlabel('Time [days]', fontsize=12, weight='bold')
        axes[0].set_ylabel('Value [USD]', fontsize=12, weight='bold')
        axes[0].grid(True)
        
        # Future predictions
        axes[1].plot(future_predictions, label='Predicted Future Prices', color='red', alpha=0.8)
        axes[1].legend()
        axes[1].set_title(f'{symbol} 3-Month Price Forecast', weight='bold')
        axes[1].set_xlabel('Time [days]', fontsize=12, weight='bold')
        axes[1].set_ylabel('Value [USD]', fontsize=12, weight='bold')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{CommonConsts.IMG_FOLDER}\\stock_rnn_predictions\\{symbol}_prediction.jpg')
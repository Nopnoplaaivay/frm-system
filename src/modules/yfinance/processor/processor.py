import pandas as pd
import numpy as np

class YfinanceProcessor:
    @classmethod
    def process_price_data(cls, df: pd.DataFrame) -> pd.DataFrame:
        processed_df = df.dropna(axis=0, how='any')
        processed_df = processed_df[[col for col in processed_df.columns if 'close' in col]]
        processed_df.columns = [col.split('_')[0] for col in processed_df.columns]
        return processed_df


    # @staticmethod
    # def build_price_matrix(processed_df: pd.DataFrame) -> pd.DataFrame:
    #     matrix = processed_df[[col for col in processed_df.columns if 'close' in col]].copy()
    #     return matrix
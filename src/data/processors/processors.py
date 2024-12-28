import numpy as np
import pandas as pd

class Processors:

    @staticmethod
    def transform(df):
        """
        This function is used to transform the data frame
        :param df: Data frame
        :return: Transformed data frame
        """
        df = df.dropna(axis=0)
        df = df.drop_duplicates()
        for col in df.columns:
            if col != 'Date':
                df[col] = df[col].str.replace(',', '').astype(float)


        return df
        
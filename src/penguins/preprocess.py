from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(df: pd.DataFrame, train_size: float = 0.7, test_size: float = 0.5,) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the data into train, validation and test sets.
    The default split is 70% train, 15% validation, 15% test.
    
    :param df: Input DataFrame.
    :param train_size: Size of the train set.
    :param test_size: Size of the test set from the remaining data after the train split.
    
    :return: Tuple of DataFrames (train, validation, test).
    """
    df_train, df_valid_test = train_test_split(df, train_size=train_size)
    df_validation, df_test = train_test_split(df_valid_test, test_size=test_size)

    return (df_train, df_validation, df_test)


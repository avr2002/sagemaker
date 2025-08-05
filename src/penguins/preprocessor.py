from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import (
    ColumnTransformer,
    make_column_selector,
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from penguins.consts import SAGEMAKER_PROCESSING_DIR
from penguins.load_data import load_data
from penguins.save import (
    save_baseline_data,
    save_processing_pipeline_model,
    save_split_data,
)


def preprocess(base_processing_directory: Union[str, Path] = SAGEMAKER_PROCESSING_DIR) -> None:
    """
    Process the data by loading it, splitting it into train, validation and test sets, and saving the processed data.

    :param base_processing_directory: Base directory where the raw data is stored.
        defaults to '/opt/ml/processing'.
    """
    # Load the raw data
    df = load_data(base_processing_directory)

    # Split the data into train, validation and test sets
    df_train, df_validation, df_test = split_data(df, train_size=0.7, test_size=0.5)

    # Save the train and test data as baselines
    save_baseline_data(data=df_train, dataset_type="train", target_column="species")
    save_baseline_data(data=df_test, dataset_type="test", target_column=None)

    # Get the features and target transformers
    features_transformer, target_transformer = preprocess_pipeline()

    # Apply Ordinal Encoding to the target variable
    # We apply fit_transform on the train set and transform on the validation and test sets
    y_train = target_transformer.fit_transform(
        np.array(df_train.species.values).reshape(-1, 1),
    )
    y_validation = target_transformer.transform(np.array(df_validation.species.values).reshape(-1, 1))
    y_test = target_transformer.transform(np.array(df_test.species.values).reshape(-1, 1))

    # Drop the target variable from the features
    df_train = df_train.drop("species", axis=1)
    df_validation = df_validation.drop("species", axis=1)
    df_test = df_test.drop("species", axis=1)

    # Apply the preprocessing pipeline to the features
    # We apply fit_transform on the train set and transform on the validation and test sets
    X_train = features_transformer.fit_transform(df_train)
    X_validation = features_transformer.transform(df_validation)
    X_test = features_transformer.transform(df_test)

    # Save the split data
    save_split_data(X_train, y_train, X_validation, y_validation, X_test, y_test)

    # Save the preprocessing pipeline model
    save_processing_pipeline_model(features_transformer, target_transformer)


def preprocess_pipeline() -> Tuple[ColumnTransformer, ColumnTransformer]:
    """
    Return the preprocessing pipeline for the features and target variables respectively.

    :return: Tuple of ColumnTransformers (features_transformer, target_transformer).
    """

    target_transformer = ColumnTransformer(transformers=[("species", OrdinalEncoder(), [0])])

    # Impute the missing numeric values with their respective mean and scale the data
    numeric_transformer = make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler(),
    )

    # Impute the most frequent value for the categorical variables and one-hot encode it
    categorical_transformer = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(),
    )

    # @TODO: Later read the feature names & target name from config file
    features_transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, make_column_selector(dtype_exclude="object")),
            # Select the 'island' column and apply the categorical transformer
            ("categorical", categorical_transformer, ["island"]),
            # We're dropping 'sex' column as it does not have any predictive power(see EDA).
        ],
    )

    return features_transformer, target_transformer


def split_data(
    df: pd.DataFrame,
    train_size: float = 0.7,
    test_size: float = 0.5,
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the data into train, validation and test sets. The default split is 70% train, 15% validation, 15% test.

    :param df: Input DataFrame.
    :param train_size: Size of the train set.
    :param test_size: Size of the test set from the remaining data after the train split.

    :return: Tuple of DataFrames (train, validation, test).
    """
    df_train, df_valid_test = train_test_split(df, train_size=train_size, random_state=random_state)
    df_validation, df_test = train_test_split(df_valid_test, test_size=test_size, random_state=random_state)

    return (df_train, df_validation, df_test)


if __name__ == "__main__":
    preprocess(base_processing_directory=SAGEMAKER_PROCESSING_DIR)
    print("Data preprocessing completed successfully.")

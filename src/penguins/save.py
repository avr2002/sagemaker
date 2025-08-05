import tarfile
import tempfile
from pathlib import Path
from typing import Literal, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from penguins.consts import SAGEMAKER_PROCESSING_DIR


def save_processing_pipeline_model(
    features_transformer: ColumnTransformer,
    target_transformer: ColumnTransformer,
    base_processing_directory: Union[str, Path] = SAGEMAKER_PROCESSING_DIR,
) -> None:
    """
    Save the pre-processing pipeline model to disk in SageMaker Processing directory.

    :param features_transformer(ColumnTransformer): The features transformer pipeline.
    :param target_transformer(ColumnTransformer): The target transformer pipeline.
    :param base_processing_directory(str | Path): Base directory where the pre-processing data is stored.
                                                  Defaults to '/opt/ml/processing'.

    :return: None
    """
    # model path
    base_processing_directory = Path(base_processing_directory)
    model_path = base_processing_directory / "model"
    model_path.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        joblib.dump(features_transformer, Path(temp_dir) / "features_transformer.joblib")
        joblib.dump(target_transformer, Path(temp_dir) / "target_transformer.joblib")

        # Zip and move the files to the base processing directory
        with tarfile.open(model_path / "model.tar.gz", "w:gz") as tar:
            tar.add(Path(temp_dir) / "features_transformer.joblib", arcname="features_transformer.joblib")
            tar.add(Path(temp_dir) / "target_transformer.joblib", arcname="target_transformer.joblib")


def save_split_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    base_processing_directory: Union[str, Path] = SAGEMAKER_PROCESSING_DIR,
) -> None:
    """
    This function concatenates the transformed features and the target variable, and
    saves each one of the split sets to disk in SageMaker Processing directory.

    Args:
        X_train (np.ndarray): The training features.
        y_train (np.ndarray): The training target variable.
        X_validation (np.ndarray): The validation features.
        y_validation (np.ndarray): The validation target variable.
        X_test (np.ndarray): The test features.
        y_test (np.ndarray): The test target variable.
        base_processing_directory (str | Path): Base directory where the pre-processing data is stored.
                                                Defaults to '/opt/ml/processing'.

    Returns:
        None
    """
    # Construct the split data path
    base_processing_directory = Path(base_processing_directory)
    train_path = base_processing_directory / "train"
    validation_path = base_processing_directory / "validation"
    test_path = base_processing_directory / "test"

    train_path.mkdir(parents=True, exist_ok=True)
    validation_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    # Concatenate the features and target variable
    train = np.concatenate((X_train, y_train), axis=1)
    validation = np.concatenate((X_validation, y_validation), axis=1)
    test = np.concatenate((X_test, y_test), axis=1)

    # Save the data as CSV
    pd.DataFrame(train).to_csv(train_path / "train.csv", header=False, index=False)
    pd.DataFrame(validation).to_csv(validation_path / "validation.csv", header=False, index=False)
    pd.DataFrame(test).to_csv(test_path / "test.csv", header=False, index=False)


def save_baseline_data(
    data: pd.DataFrame,
    dataset_type: Literal["train", "test"],
    target_column: Optional[str] = "species",
    base_processing_directory: Union[str, Path] = SAGEMAKER_PROCESSING_DIR,
) -> None:
    """
    Save the untransformed data (train or test) to disk as a baseline.
    We will need the training/test data to compute a baseline to determine
    the quality of the model predictions when deployed.

    Args:
        data (pd.DataFrame): The DataFrame to save.
        dataset_type (Literal["train", "test"]): Specify 'train' or 'test' to determine the dataset type to save.
        target_column (str | None): The target column to drop for the train baseline. Default is 'species'.
        base_processing_directory (str | Path): Base directory where the pre-processing data is stored.
                                                Defaults to '/opt/ml/processing'.

    Returns:
        None
    """
    # Validate dataset_type
    if dataset_type not in {"train", "test"}:
        raise ValueError("dataset_type must be 'train' or 'test'.")

    # Validate target_column if dataset_type is 'train'
    if dataset_type == "train" and target_column is None:
        raise ValueError("target_column must be specified for the 'train' dataset type.")

    # Construct the baseline path
    baseline_path = Path(base_processing_directory) / f"{dataset_type}-baseline"
    baseline_path.mkdir(parents=True, exist_ok=True)

    df_copy = data.copy().dropna()

    # Handle train-specific processing
    if dataset_type == "train":
        # To compute the data quality baseline, we don't need the
        # target variable, so we'll drop it from the dataframe.
        df_copy = df_copy.drop(target_column, axis=1)

    # Handle test-specific processing (no header)
    # We'll use the test baseline to generate predictions later, and we can't have a header line
    # because the model won't be able to make a prediction for it. So, when saving the test data,
    # we'll exclude the headers.
    include_header = dataset_type == "train"  # False for test, True for train

    # Save to CSV
    df_copy.to_csv(baseline_path / f"{dataset_type}-baseline.csv", header=include_header, index=False)

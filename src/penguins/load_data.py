"""Load Data Step."""

from pathlib import Path
from typing import Optional, Union

import pandas as pd

from penguins.consts import SAGEMAKER_PROCESSING_DIR

# import boto3
# from botocore.exceptions import ClientError
# try:
#     from mypy_boto3_s3 import S3Client
#     from mypy_boto3_s3.type_defs import (
#         GetObjectOutputTypeDef,
#         HeadObjectOutputTypeDef,
#     )
# except ImportError:
#     ...


def load_data(
    base_processing_directory: Union[str, Path] = SAGEMAKER_PROCESSING_DIR,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load CSV data inside SageMaker Processing container.
    This function reads every CSV file available and
    concatenates them into a single dataframe.

    :param base_processing_directory: Base directory where the pre-processing data is stored.
                                      Defaults to '/opt/ml/processing'.
    :param random_state: Random seed for shuffling the data.

    :return: Pandas DataFrame.
    """
    base_processing_directory = Path(base_processing_directory)

    input_directory = base_processing_directory / "input"
    csv_files = list(input_directory.glob("*.csv"))

    if len(csv_files) == 0:
        raise ValueError(f"No CSV files found in {input_directory.as_posix()}")

    raw_data = [pd.read_csv(file) for file in csv_files]
    df = pd.concat(raw_data)

    # Shuffle the data
    return df.sample(frac=1, random_state=random_state)


# def load_data_from_disk(file_path: Union[Path, str]) -> pd.DataFrame:
#     """
#     Load data from disk into a pandas DataFrame.

#     :param file_path: Path to the data file.
#     :return: Pandas DataFrame.
#     """
#     if Path(file_path).exists():
#         return pd.read_csv(file_path)
#     else:
#         raise FileNotFoundError(f"File not found: {file_path}")


# def load_data_from_s3(bucket: str, object_key: str, s3_client: Optional["S3Client"] = None) -> pd.DataFrame:
#     """
#     Load data from S3 into a pandas DataFrame.

#     :param bucket: S3 bucket name.
#     :param object_key: S3 key.

#     :return: Pandas DataFrame.
#     """
#     if object_exists_in_s3(bucket, object_key):
#         s3_client = s3_client or boto3.client("s3")
#         response: "GetObjectOutputTypeDef" = s3_client.get_object(Bucket=bucket, Key=object_key)
#         return pd.read_csv(response["Body"])
#     else:
#         raise FileNotFoundError(f"File not found: s3://{bucket}/{object_key}")


# def object_exists_in_s3(  # type: ignore
#     bucket_name: str,
#     object_key: str,
#     s3_client: Optional["S3Client"] = None,
# ) -> bool:
#     """
#     Check if an object exists in the S3 bucket using head_object.

#     :param bucket_name: Name of the S3 bucket.
#     :param object_key: Key of the object to check.
#     :param s3_client: Optional S3 client to use. If not provided, a new client will be created.

#     :return: True if the object exists, False otherwise.
#     """
#     try:
#         s3_client = s3_client or boto3.client("s3")
#         response: "HeadObjectOutputTypeDef" = s3_client.head_object(Bucket=bucket_name, Key=object_key)
#         if response:
#             return True
#     except ClientError as err:
#         error_code = err.response.get("Error", {}).get("Code", "")
#         if error_code == "404":
#             return False
#         raise

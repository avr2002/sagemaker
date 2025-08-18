"""Lambda function to get baseline accuracy from the latest registered model."""

import json
import os
from typing import List

import boto3

try:
    from mypy_boto3_s3.client import S3Client
    from mypy_boto3_s3.type_defs import GetObjectOutputTypeDef
    from mypy_boto3_sagemaker.client import SageMakerClient
    from mypy_boto3_sagemaker.type_defs import (
        DescribeModelPackageOutputTypeDef,
        ListModelPackagesOutputTypeDef,
        ModelPackageSummaryTypeDef,
    )
except ImportError:
    print("mypy_boto3_sagemaker is not installed")


region = os.environ["AWS_REGION"]


def get_baseline_accuracy_of_last_registered_model(model_package_group_name: str) -> float:
    """
    Get the baseline accuracy from the last registered model in a SageMaker model package group.

    :param model_package_group_name: The name of the model package group.
    :return: The baseline accuracy of the last registered model.
    """
    sm_client: "SageMakerClient" = boto3.client("sagemaker", region_name=region)
    baseline_accuracy = 0.0

    try:
        # List model packages in the group, sorted by creation time (newest first)
        # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker/client/list_model_packages.html
        response: "ListModelPackagesOutputTypeDef" = sm_client.list_model_packages(
            ModelPackageGroupName=model_package_group_name,
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=1,  # Only get the latest one
        )

        model_packages: "List[ModelPackageSummaryTypeDef]" = response.get("ModelPackageSummaryList", [])
        if not model_packages:
            print(f"No model packages found in group: {model_package_group_name}")
            return baseline_accuracy

        # Get the latest model package ARN
        latest_model_package_arn = model_packages[0]["ModelPackageArn"]
        print(f"Found latest model package: {latest_model_package_arn}")

        # Describe the model package to get its metrics
        # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker/client/describe_model_package.html
        model_package_details: "DescribeModelPackageOutputTypeDef" = sm_client.describe_model_package(
            ModelPackageName=latest_model_package_arn
        )

        # Extract accuracy from model metrics
        # model_metrics = model_package_details.get("ModelMetrics", {})
        # model_quality = model_metrics.get("ModelQuality", {})
        # statistics = model_quality.get("Statistics", {})
        # s3_uri = statistics.get("S3Uri", "")

        # The statistics are stored in S3 as JSON, we need to download and parse them
        s3_uri = model_package_details["ModelMetrics"]["ModelQuality"]["Statistics"]["S3Uri"]
        baseline_accuracy = get_baseline_accuracy_from_s3(s3_uri)
        return baseline_accuracy
    except sm_client.exceptions.ResourceNotFound:
        print(f"Model package group not found: {model_package_group_name}")
        return baseline_accuracy


def get_baseline_accuracy_from_s3(s3_uri: str) -> float:
    try:
        # Parse S3 URI to get bucket and key
        s3_parts = s3_uri.replace("s3://", "").split("/", 1)
        bucket_name = s3_parts[0]
        object_key = s3_parts[1]

        # Download and parse the metrics file
        s3_client: "S3Client" = boto3.client("s3", region_name=region)
        response: "GetObjectOutputTypeDef" = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        metrics_content = response["Body"].read().decode("utf-8")
        metrics_data = json.loads(metrics_content)

        # Extract accuracy value
        accuracy = metrics_data.get("metrics", {}).get("accuracy", {}).get("value", 0.0)
        print(f"Found baseline accuracy: {accuracy}")
        return float(accuracy)
    # When the pipeline runs for very first time we won't have any previous evaluation reports stored in S3
    except s3_client.exceptions.NoSuchKey:
        print(f"No such key found in S3: {s3_uri}, returning 0.0 as baseline accuracy!")
        return 0.0

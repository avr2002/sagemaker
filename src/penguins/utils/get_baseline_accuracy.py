"""Lambda function to get baseline accuracy from the latest registered model."""

import json
import os
from typing import Any, Dict, List

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


def get_baseline_accuracy_of_last_registered_model(model_package_group_name: str) -> float:
    """
    Get the baseline accuracy from the last registered model in a SageMaker model package group.

    :param model_package_group_name: The name of the model package group.
    :return: The baseline accuracy of the last registered model.
    """
    region = os.environ["AWS_REGION"]
    sm_client: "SageMakerClient" = boto3.client("sagemaker", region_name=region)

    try:
        # List model packages in the group, sorted by creation time (newest first)
        response: "ListModelPackagesOutputTypeDef" = sm_client.list_model_packages(
            ModelPackageGroupName=model_package_group_name,
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=1,  # Only get the latest one
        )

        model_packages: "List[ModelPackageSummaryTypeDef]" = response.get("ModelPackageSummaryList", [])

        if not model_packages:
            print(f"No model packages found in group: {model_package_group_name}")
            return 0.0

        # Get the latest model package ARN
        latest_model_package_arn = model_packages[0]["ModelPackageArn"]
        print(f"Found latest model package: {latest_model_package_arn}")

        # Describe the model package to get its metrics
        model_package_details: "DescribeModelPackageOutputTypeDef" = sm_client.describe_model_package(
            ModelPackageName=latest_model_package_arn
        )

        # Extract accuracy from model metrics
        model_metrics = model_package_details.get("ModelMetrics", {})
        model_quality = model_metrics.get("ModelQuality", {})
        statistics = model_quality.get("Statistics", {})
        # model_package_details["ModelMetrics"]["ModelQuality"]["Statistics"]["S3Uri"]

        if statistics:
            # The statistics are stored in S3, we need to download and parse them
            s3_uri = statistics.get("S3Uri", "")
            if s3_uri:
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
    except sm_client.exceptions.ResourceNotFound:
        print(f"Model package group not found: {model_package_group_name}")
        return 0.0
    # except Exception as e:
    #     print(f"Error getting baseline accuracy: {str(e)}")
    #     return 0.0
    print("Error getting baseline accuracy, returning 0.0")
    return 0.0


# def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
#     """
#     Lambda handler to get the accuracy of the latest registered model.

#     Args:
#         event: Lambda event containing model_package_group_name and region
#         context: Lambda context

#     Returns:
#         Dictionary containing the baseline accuracy
#     """
#     try:
#         model_package_group_name = event["model_package_group_name"]
#         region = event.get("region", "us-east-1")

#         # Initialize SageMaker client
#         sm_client: "SageMakerClient" = boto3.client("sagemaker", region_name=region)

#         try:
#             # List model packages in the group, sorted by creation time (newest first)
#             # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker/client/list_model_packages.html
#             response: "ListModelPackagesOutputTypeDef" = sm_client.list_model_packages(
#                 ModelPackageGroupName=model_package_group_name,
#                 SortBy="CreationTime",
#                 SortOrder="Descending",
#                 MaxResults=1,  # Only get the latest one
#                 ModelApprovalStatus="Approved",  # Only consider approved models
#             )

#             model_packages: "List[ModelPackageSummaryTypeDef]" = response.get("ModelPackageSummaryList", [])

#             if not model_packages:
#                 print(f"No approved model packages found in group: {model_package_group_name}")
#                 return {"baseline_accuracy": 0.0}

#             # Get the latest model package ARN
#             latest_model_package_arn = model_packages[0]["ModelPackageArn"]
#             print(f"Found latest model package: {latest_model_package_arn}")

#             # Describe the model package to get its metrics
#             # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker/client/describe_model_package.html
#             model_package_details: "DescribeModelPackageOutputTypeDef" = sm_client.describe_model_package(
#                 ModelPackageName=latest_model_package_arn
#             )
#             # Extract accuracy from model metrics
#             model_metrics = model_package_details.get("ModelMetrics", {})
#             model_quality = model_metrics.get("ModelQuality", {})
#             statistics = model_quality.get("Statistics", {})
#             # model_package_details["ModelMetrics"]["ModelQuality"]["Statistics"]["S3Uri"]

#             if statistics:
#                 # The statistics are stored in S3, we need to download and parse them
#                 s3_uri = statistics.get("S3Uri", "")
#                 if s3_uri:
#                     # Parse S3 URI to get bucket and key
#                     s3_parts = s3_uri.replace("s3://", "").split("/", 1)
#                     bucket_name = s3_parts[0]
#                     object_key = s3_parts[1]

#                     # Download and parse the metrics file
#                     s3_client: "S3Client" = boto3.client("s3", region_name=region)
#                     response: "GetObjectOutputTypeDef" = s3_client.get_object(Bucket=bucket_name, Key=object_key)
#                     metrics_content = response["Body"].read().decode("utf-8")
#                     metrics_data = json.loads(metrics_content)

#                     # Extract accuracy value
#                     accuracy = metrics_data.get("metrics", {}).get("accuracy", {}).get("value", 0.0)
#                     print(f"Found baseline accuracy: {accuracy}")
#                     return {"baseline_accuracy": float(accuracy)}

#             print("No metrics found in the latest model package")
#             return {"baseline_accuracy": 0.0}

#         except sm_client.exceptions.ResourceNotFound:
#             print(f"Model package group not found: {model_package_group_name}")
#             return {"baseline_accuracy": 0.0}

#     except Exception as e:
#         print(f"Error getting baseline accuracy: {str(e)}")
#         # Return a very low baseline so the current model will likely be registered
#         return {"baseline_accuracy": 0.0}

#!/usr/bin/env python3
"""Deploy endpoint from registered model."""

import os

import boto3

# import argparse


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--inference-image", required=True, help="ECR image URI for inference")
    # args = parser.parse_args()

    sagemaker = boto3.client("sagemaker", region_name=os.environ.get("AWS_REGION", "us-east-1"))

    # Get the latest approved model from model registry
    model_package_group_name = os.environ["MODEL_PACKAGE_GROUP_NAME"]

    # List model packages and get the latest approved one
    response = sagemaker.list_model_packages(
        ModelPackageGroupName=model_package_group_name,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )

    if not response["ModelPackageSummaryList"]:
        raise ValueError("No approved model found in registry")

    model_package_arn = response["ModelPackageSummaryList"][0]["ModelPackageArn"]

    # Create a SageMaker Model from the model package
    import time

    model_name = f"penguins-model-{int(time.time())}"

    sagemaker.create_model(
        ModelName=model_name,
        Containers=[
            {
                "ModelPackageName": model_package_arn,
                # "Image": args.inference_image
                # (ValidationException) when calling the CreateModel operation: Specify one of either Image and Model Data URL or ModelPackageName in container definition.
            }
        ],
        ExecutionRoleArn=os.environ["SAGEMAKER_EXECUTION_ROLE"],
    )

    # Create endpoint configuration
    endpoint_config_name = f"penguins-endpoint-config-{int(time.time())}"
    endpoint_name = "penguins-endpoint"

    try:
        sagemaker.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    "ModelName": model_name,  # Use the model name, not ARN
                    "VariantName": "primary",
                    "InstanceType": "ml.m5.2xlarge",
                    "InitialInstanceCount": 1,
                    "InitialVariantWeight": 1.0,
                }
            ],
            # ProductionVariants=[
            #     {
            #         "ModelName": model_name,
            #         "VariantName": "AllTraffic",
            #         "ServerlessConfig": {
            #             "MemorySizeInMB": 4096,
            #             "MaxConcurrency": 20,
            #             # "ProvisionedConcurrency": 10,
            #         },
            #     }
            # ],
        )

        # Create or update endpoint
        try:
            sagemaker.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name)
            print(f"Endpoint {endpoint_name} creation started")
        except sagemaker.exceptions.ClientError as create_error:
            if "already existing endpoint" in str(create_error):
                # Update existing endpoint
                sagemaker.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name)
                print(f"Endpoint {endpoint_name} update started")
            else:
                raise

    except Exception as e:
        if "already exists" in str(e):
            print(f"Endpoint config {endpoint_config_name} already exists")
        else:
            raise


if __name__ == "__main__":
    main()

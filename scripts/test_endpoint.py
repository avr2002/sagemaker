#!/usr/bin/env python3
"""Test the deployed SageMaker endpoint."""

import json

import boto3
import pandas as pd
from rich import print


def test_endpoint_csv():
    """Test endpoint with CSV format"""
    # Initialize SageMaker runtime client
    runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")

    # Sample penguin data (same format as training data)
    # ['island', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
    test_data = {
        "island": ["Torgersen", "Dream", "Biscoe"],
        "culmen_length_mm": [39.1, 46.5, 50.0],
        "culmen_depth_mm": [18.7, 17.9, 19.5],
        "flipper_length_mm": [181, 192, 196],
        "body_mass_g": [3750, 4150, 3900],
    }

    # Convert to CSV format (with headers)
    df = pd.DataFrame(test_data)
    csv_data = df.to_csv(index=False, header=True)

    print("=== Testing CSV Format ===")
    print("Test data:")
    print(csv_data)

    # Invoke endpoint
    response = runtime.invoke_endpoint(EndpointName="penguins-endpoint", ContentType="text/csv", Body=csv_data)
    print(f"Response status: {response['ResponseMetadata']['HTTPStatusCode']}")

    # Parse response
    result = json.loads(response["Body"].read().decode())
    print(f"Result: {result}")

    print("Predictions:")
    for i, prediction in enumerate(result):
        species = prediction["prediction"]
        confidence = prediction["confidence"]
        probabilities = prediction["probabilities"]
        print(f"  Penguin {i+1}: {species} (confidence: {confidence:.4f})")
        print(f"    Probabilities: {probabilities}")


def test_endpoint_json():
    """Test endpoint with JSON format"""
    # Initialize SageMaker runtime client
    runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")

    # Sample penguin data as JSON payload
    json_payload = {
        "instances": [
            {
                "island": "Torgersen",
                "culmen_length_mm": 39.1,
                "culmen_depth_mm": 18.7,
                "flipper_length_mm": 181,
                "body_mass_g": 3750
            },
            {
                "island": "Dream",
                "culmen_length_mm": 46.5,
                "culmen_depth_mm": 17.9,
                "flipper_length_mm": 192,
                "body_mass_g": 4150
            },
            {
                "island": "Biscoe",
                "culmen_length_mm": 50.0,
                "culmen_depth_mm": 19.5,
                "flipper_length_mm": 196,
                "body_mass_g": 3900
            }
        ]
    }

    json_data = json.dumps(json_payload)

    print("\n=== Testing JSON Format ===")
    print("JSON payload:")
    print(json.dumps(json_payload, indent=2))

    # Invoke endpoint
    response = runtime.invoke_endpoint(
        EndpointName="penguins-endpoint",
        ContentType="application/json",
        Body=json_data
    )
    print(f"Response status: {response['ResponseMetadata']['HTTPStatusCode']}")

    # Parse response
    result = json.loads(response["Body"].read().decode())
    print(f"Result: {result}")

    print("Predictions:")
    for i, prediction in enumerate(result):
        species = prediction["prediction"]
        confidence = prediction["confidence"]
        probabilities = prediction["probabilities"]
        print(f"  Penguin {i+1}: {species} (confidence: {confidence:.4f})")
        print(f"    Probabilities: {probabilities}")


def test_single_prediction_json():
    """Test endpoint with a single penguin record in JSON format"""
    # Initialize SageMaker runtime client
    runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")

    # Single penguin record
    single_penguin = {
        "island": "Torgersen",
        "culmen_length_mm": 39.1,
        "culmen_depth_mm": 18.7,
        "flipper_length_mm": 181,
        "body_mass_g": 3750
    }

    json_data = json.dumps(single_penguin)

    print("\n=== Testing Single Prediction (JSON) ===")
    print("Single penguin data:")
    print(json.dumps(single_penguin, indent=2))

    # Invoke endpoint
    response = runtime.invoke_endpoint(
        EndpointName="penguins-endpoint",
        ContentType="application/json",
        Body=json_data
    )
    print(f"Response status: {response['ResponseMetadata']['HTTPStatusCode']}")

    # Parse response
    result = json.loads(response["Body"].read().decode())
    print(f"Result: {result}")

    # Handle single prediction result
    if isinstance(result, list) and len(result) == 1:
        prediction = result[0]
    elif isinstance(result, dict):
        prediction = result
    else:
        print(f"Unexpected result format: {result}")
        return

    species = prediction["prediction"]
    confidence = prediction["confidence"]
    probabilities = prediction["probabilities"]
    print(f"Prediction: {species} (confidence: {confidence:.4f})")
    print(f"Probabilities: {probabilities}")


def test_endpoint():
    """Test endpoint with multiple formats"""
    test_endpoint_csv()
    test_endpoint_json()
    test_single_prediction_json()


if __name__ == "__main__":
    test_endpoint()

#!/usr/bin/env python3
"""Test the deployed SageMaker endpoint."""

import json

import boto3
import pandas as pd
from rich import print


def test_endpoint():
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

    # Convert to CSV format (without headers, as expected by model)
    df = pd.DataFrame(test_data)
    csv_data = df.to_csv(index=False, header=True)

    print("Test data:")
    print(csv_data)

    # Invoke endpoint
    response = runtime.invoke_endpoint(EndpointName="penguins-endpoint", ContentType="text/csv", Body=csv_data)
    print(f"{response=}")

    # Parse response
    result = json.loads(response["Body"].read().decode())
    print(f"{result=}")

    print("Predictions:")
    for i, j in enumerate(result):
        species = j["prediction"]
        confidence = j["confidence"]
        probabilities = j["probabilities"]
        print(f"  Species: {species}, Confidence: {confidence:.4f}, Probabilities: {probabilities}")


if __name__ == "__main__":
    test_endpoint()

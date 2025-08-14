import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from packaging import version
from sklearn.metrics import accuracy_score
from tensorflow import keras

from penguins.consts import SAGEMAKER_PROCESSING_DIR


def evaluate(
    model_path: str | Path,
    test_path: str | Path,
    output_path: str | Path,
) -> None:
    """Model evaluation Script.

    :param model_path: The path to the trained model directory.
    :param test_path: The path to the test data directory.
    :param output_path: The path to the output directory where we will save evaluation reports.
    """
    X_test = pd.read_csv(Path(test_path) / "test.csv")
    y_test = X_test[X_test.columns[-1]]
    X_test = X_test.drop(X_test.columns[-1], axis=1)

    # Let's now extract the model package so we can load it in memory.
    with tarfile.open(Path(model_path) / "model.tar.gz") as tar:
        tar.extractall(path=Path(model_path))

    # Use the same version-based logic as in the training script
    model_filepath = (
        Path(model_path) / "001"
        if version.parse(keras.__version__) < version.parse("3")
        else Path(model_path) / "penguins.keras"
    )

    model = keras.models.load_model(model_filepath)

    predictions = np.argmax(model.predict(X_test), axis=-1)
    accuracy = accuracy_score(y_true=y_test, y_pred=predictions)
    print(f"Test accuracy: {accuracy}")

    # Let's create an evaluation report using the model accuracy.
    # We will save this evaluation report as a JSON file, what Sagemaker calls a "property file"
    # And will use this property file in the next steps of the pipeline.
    # ref: https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-propertyfile.html
    evaluation_report = {
        "metrics": {
            "accuracy": {"value": accuracy},
            # "precision": {"value": precision},
        },
    }

    Path(output_path).mkdir(parents=True, exist_ok=True)
    with open(Path(output_path) / "evaluation.json", "w") as f:
        f.write(json.dumps(evaluation_report))


if __name__ == "__main__":
    print("Starting Model Evaluation...")

    evaluate(
        model_path=SAGEMAKER_PROCESSING_DIR / "model/",  # /opt/ml/processing/model
        test_path=SAGEMAKER_PROCESSING_DIR / "test/",
        output_path=SAGEMAKER_PROCESSING_DIR / "evaluation/",
    )

    print("Model Evaluation Completed!")

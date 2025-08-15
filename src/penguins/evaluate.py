import json
import os
import tarfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from comet_ml import Experiment
from packaging import version
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow import keras

from penguins.consts import SAGEMAKER_PROCESSING_DIR


def evaluate(
    model_path: str | Path,
    test_path: str | Path,
    output_path: str | Path,
    experiment: Optional[Experiment] = None,
) -> None:
    """Model evaluation Script.

    :param model_path: The path to the trained model directory.
    :param test_path: The path to the test data directory.
    :param output_path: The path to the output directory where we will save evaluation reports.
    :param experiment: Comet ML experiment object.
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

    # Compute confusion matrix
    cm = confusion_matrix(y_true=y_test, y_pred=predictions)
    print(f"Confusion Matrix:\n{cm}")

    # Log metrics and confusion matrix to Comet
    if experiment:
        experiment.log_metric("test_accuracy", accuracy)
        experiment.log_dataset_hash(X_test)
        experiment.log_confusion_matrix(
            y_true=y_test.astype(int),
            y_predicted=predictions.astype(int),
            title="Test Set Confusion Matrix",
        )

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

    # Let's create a Comet experiment to log the metrics of this evaluation job.
    comet_api_key = os.environ.get("COMET_API_KEY", None)
    comet_project_name = os.environ.get("COMET_PROJECT_NAME", None)

    # COMET WARNING: To get all data logged automatically, import comet_ml before the following modules: tensorboard, keras, tensorflow
    experiment = (
        Experiment(
            project_name=comet_project_name,
            api_key=comet_api_key,
            auto_metric_logging=True,
            auto_param_logging=True,
            log_code=True,
        )
        if comet_api_key and comet_project_name
        else None
    )

    # We want to use a descriptive name for the evaluation experiment
    if experiment:
        experiment.set_name("model-evaluation")
        experiment.add_tag("evaluation")

    evaluate(
        model_path=SAGEMAKER_PROCESSING_DIR / "model/",  # /opt/ml/processing/model
        test_path=SAGEMAKER_PROCESSING_DIR / "test/",
        output_path=SAGEMAKER_PROCESSING_DIR / "evaluation/",
        experiment=experiment,
    )

    print("Model Evaluation Completed!")

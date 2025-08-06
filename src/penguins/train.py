"""Train Script."""

import argparse
import json
import os
import tarfile
from pathlib import Path
from typing import Optional, Union

import keras
import numpy as np
import pandas as pd
from comet_ml import Experiment
from keras import Input
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from packaging import version
from sklearn.metrics import accuracy_score


def train(
    model_directory: Union[str, Path],
    train_path: Union[str, Path],
    validation_path: Union[str, Path],
    pipeline_path: Union[str, Path],
    experiment: Optional[Experiment],
    epochs: int = 50,
    batch_size: int = 32,
):
    print(f"Keras version: {keras.__version__}")

    X_train = pd.read_csv(Path(train_path) / "train.csv")
    y_train = X_train[X_train.columns[-1]]
    X_train = X_train.drop(X_train.columns[-1], axis=1)

    X_validation = pd.read_csv(Path(validation_path) / "validation.csv")
    y_validation = X_validation[X_validation.columns[-1]]
    X_validation = X_validation.drop(X_validation.columns[-1], axis=1)

    model = Sequential(
        [
            Input(shape=(X_train.shape[1],)),
            Dense(10, activation="relu"),
            Dense(8, activation="relu"),
            Dense(3, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=SGD(learning_rate=0.01),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_validation, y_validation),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
    )

    predictions = np.argmax(model.predict(X_validation), axis=-1)
    val_accuracy = accuracy_score(y_true=y_validation, y_pred=predictions)
    print(f"Validation accuracy: {val_accuracy}")

    # Starting on version 3, Keras changed the model saving format.
    # Since we are running the training script using two different versions
    # of Keras, we need to check to see which version we are using and save
    # the model accordingly.
    model_filepath = (
        Path(model_directory) / "001"
        if version.parse(keras.__version__) < version.parse("3")
        else Path(model_directory) / "penguins.keras"
    )

    model.save(model_filepath)

    # Let's save the transformation pipelines inside the
    # model directory so they get bundled together.
    with tarfile.open(Path(pipeline_path) / "model.tar.gz", mode="r:gz") as tar:
        tar.extractall(model_directory)

    if experiment:
        experiment.log_parameters(
            {
                "epochs": epochs,
                "batch_size": batch_size,
            }
        )
        experiment.log_dataset_hash(X_train)
        experiment.log_confusion_matrix(
            y_true=y_validation.astype(int),
            y_predicted=predictions.astype(int),
        )
        experiment.log_model("penguins", model_filepath.as_posix())


if __name__ == "__main__":
    print("Starting training script...")
    # Any hyperparameters provided by the training job are passed to
    # the entry point as script arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    args, _ = parser.parse_known_args()

    # Let's create a Comet experiment to log the metrics and parameters
    # of this training job.
    comet_api_key = os.environ.get("COMET_API_KEY", None)
    comet_project_name = os.environ.get("COMET_PROJECT_NAME", None)

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

    training_env = json.loads(os.environ.get("SM_TRAINING_ENV", {}))
    job_name = training_env.get("job_name", None) if training_env else None

    # We want to use the SageMaker's training job name as the name
    # of the experiment so we can easily recognize it.
    if job_name and experiment:
        experiment.set_name(job_name)

    train(
        # This is the location where we need to save our model.
        # SageMaker will create a model.tar.gz file with anything
        # inside this directory when the training script finishes.
        model_directory=os.environ["SM_MODEL_DIR"],
        # SageMaker creates one channel for each one of the inputs
        # to the Training Step.
        train_path=os.environ["SM_CHANNEL_TRAIN"],
        validation_path=os.environ["SM_CHANNEL_VALIDATION"],
        pipeline_path=os.environ["SM_CHANNEL_PIPELINE"],
        experiment=experiment,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print("Training script completed successfully!")

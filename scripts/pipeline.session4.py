"""
Setup Sagemaker Preprocessing Job Pipeline.

https://github.com/aws/amazon-sagemaker-examples/tree/main
"""

# A processing step requires a processor, a Python script that defines
# the processing code, outputs for processing, and job arguments.
import os
from pathlib import Path

import sagemaker
from dotenv import load_dotenv
from sagemaker.inputs import TrainingInput
from sagemaker.processing import FrameworkProcessor, ProcessingInput, ProcessingOutput
from sagemaker.tensorflow.estimator import TensorFlow
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import LocalPipelineSession, PipelineSession

# from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.pipeline_definition_config import PipelineDefinitionConfig
from sagemaker.workflow.steps import CacheConfig, ProcessingStep, TrainingStep

from penguins.consts import BUCKET, S3_LOCATION, SAGEMAKER_EXECUTION_ROLE, SAGEMAKER_PROCESSING_DIR, SRC_DIR
from penguins.utils import build_docker_image

THIS_DIR = Path(__file__).parent

# Load environment variables from .env file
load_dotenv()
env_vars = {
    "COMET_API_KEY": os.getenv("COMET_API_KEY", ""),
    "COMET_PROJECT_NAME": os.getenv("COMET_PROJECT_NAME", ""),
    "S3_BUCKET_NAME": os.getenv("S3_BUCKET_NAME", ""),
    "SAGEMAKER_EXECUTION_ROLE": os.getenv("SAGEMAKER_EXECUTION_ROLE", ""),
    "LOCAL_MODE": os.getenv("LOCAL_MODE", ""),
}

# Create a local Sagemaker session
sagemaker_session = (
    LocalPipelineSession(default_bucket=BUCKET)
    if env_vars["LOCAL_MODE"] == "true"
    else PipelineSession(default_bucket=BUCKET)
)
# https://docs.aws.amazon.com/sagemaker/latest/dg/notebooks-available-instance-types.html
# locally instance_type can also be "local_gpu"
instance_type = "local" if env_vars["LOCAL_MODE"] == "true" else "ml.m5.2xlarge"  # "ml.m5.xlarge"

# WARNING:sagemaker.workflow.utilities:Popping out 'ProcessingJobName' from the pipeline definition
# by default since it will be overridden at pipeline execution time.
# Please utilize the PipelineDefinitionConfig to persist this field in the pipeline definition if desired.

# WARNING:sagemaker.workflow.utilities:Popping out 'TrainingJobName' from the pipeline definition
# by default since it will be overridden at pipeline execution time.
# Please utilize the PipelineDefinitionConfig to persist this field in the pipeline definition if desired.
pipeline_definition_config = PipelineDefinitionConfig(use_custom_job_prefix=True)

# define a parameter for the input data
dataset_location = ParameterString(name="dataset-location", default_value=f"{S3_LOCATION}/data/")

# Setup Cache for the pipeline step
cache_config = CacheConfig(enable_caching=False, expire_after="15d")

# Sagemaker Processing Job
# https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_processing.html#amazon-sagemaker-processing

# ref: https://github.com/aws/amazon-sagemaker-examples/blob/main/sagemaker_processing/scikit_learn_data_processing_and_model_evaluation/scikit_learn_data_processing_and_model_evaluation.ipynb
est_cls = sagemaker.sklearn.estimator.SKLearn
# ref: https://docs.aws.amazon.com/sagemaker/latest/dg/sklearn.html
framework_version_str = "1.2-1"  # Sagemaker Scikit-learn version

framework_processor = FrameworkProcessor(
    base_job_name="data-preprocessing",
    role=SAGEMAKER_EXECUTION_ROLE,
    instance_count=1,
    instance_type=instance_type,
    estimator_cls=est_cls,
    framework_version=framework_version_str,
    sagemaker_session=sagemaker_session,
    env=env_vars,
    # py_version="py3",
)

# sklearn_processor = SKLearnProcessor(
#     base_job_name="data-preprocessing",
#     framework_version="1.2-1",
#     role=SAGEMAKER_EXECUTION_ROLE,
#     instance_type="local",
#     instance_count=1,
#     sagemaker_session=sagemaker_session,
# )

preprocessing_step = ProcessingStep(
    name="data-preprocessing",
    step_args=framework_processor.run(
        code="src/penguins/preprocessor.py",
        # source_dir="src/penguins",
        # While installing the local package (via pip install .) in requirements.txt, we need to pass the pyproject.toml and README.md files
        # Right now, we are not installing the local package because of dependency conflicts
        # dependencies=["src/penguins", "requirements.txt", "pyproject.toml", "README.md"],
        dependencies=["src/penguins", "requirements.txt"],
        inputs=[
            ProcessingInput(source=dataset_location, destination=(SAGEMAKER_PROCESSING_DIR / "input").as_posix()),
        ],
        outputs=[
            ProcessingOutput(
                output_name=output_name,
                source=f"{SAGEMAKER_PROCESSING_DIR}/{output_name}",
                destination=f"{S3_LOCATION}/preprocessing/{output_name}",
            )
            for output_name in [
                "train",
                "validation",
                "test",
                "preprocessing-pipeline",
                "train-baseline",
                "test-baseline",
            ]
        ],
    ),
    cache_config=cache_config,
)

# Create a TensorFlow estimator for the training step

# Guide:
# https://sagemaker.readthedocs.io/en/stable/
# https://sagemaker.readthedocs.io/en/stable/frameworks/tensorflow/using_tf.html
# https://sagemaker-examples.readthedocs.io/en/latest/sagemaker-script-mode/sagemaker-script-mode.html
# https://docs.aws.amazon.com/sagemaker/latest/dg/tf.html
# https://github.com/aws/sagemaker-tensorflow-training-toolkit

# Custom Docker Images:
# https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-training-container.html
# https://docs.aws.amazon.com/sagemaker/latest/dg/amazon-sagemaker-toolkits.html
# https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-custom-images.html

# Tensorflow Pre-built Containers: https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/appendix-dlc-release-notes-tensorflow.html
# https://docs.aws.amazon.com/sagemaker/latest/dg/notebooks-available-images.html
# https://github.com/aws/deep-learning-containers/blob/master/available_images.md
# image_uri = sagemaker.image_uris.retrieve(
#     framework="tensorflow",
#     region=sagemaker_session.boto_region_name,
#     version="2.12.0",
#     image_scope="training",
#     instance_type="local",
#     py_version="py310",
#     sagemaker_session=sagemaker_session,
# )
# print(f"\n\n{image_uri=}\n\n")

# To test your training pipeline locally, we need to build a custom arm compatible Docker image for M-series Mac.
# docker build -t sagemaker-tf-training-toolkit-arm64:latest containers/Dockerfile
# image_uri = "sagemaker-tf-training-toolkit-arm64:latest" if env_vars["LOCAL_MODE"] == "true" else None

if os.environ["LOCAL_MODE"] == "true":
    # image_uri = "sagemaker-tf-training-toolkit-arm64:latest"
    image_uri = build_docker_image(
        repository_name="sagemaker-tf-training-toolkit-arm64",
        dockerfile_fpath=THIS_DIR / "containers/Dockerfile",
        tag="latest",
    )
else:
    image_uri = None

# https://sagemaker.readthedocs.io/en/stable/frameworks/tensorflow/using_tf.html#id5
tf_estimator = TensorFlow(
    base_job_name="training",
    entry_point="src/penguins/train.py",
    # source_dir="src/penguins",
    dependencies=["src/penguins", "requirements.txt"],
    # SageMaker will pass these hyperparameters as arguments
    # to the entry point of the training script.
    hyperparameters={
        "epochs": 50,
        "batch_size": 32,
    },
    # SageMaker will create these environment variables on the
    # Training Job instance.
    environment=env_vars,
    # SageMaker will track these metrics as part of the experiment
    # associated to this pipeline. The metric definitions tells
    # SageMaker how to parse the values from the Training Job logs.
    metric_definitions=[
        {"Name": "loss", "Regex": "loss: ([0-9\\.]+)"},
        {"Name": "accuracy", "Regex": "accuracy: ([0-9\\.]+)"},
        {"Name": "val_loss", "Regex": "val_loss: ([0-9\\.]+)"},
        {"Name": "val_accuracy", "Regex": "val_accuracy: ([0-9\\.]+)"},
    ],
    image_uri=image_uri,
    framework_version="2.12.0",
    py_version="py310",
    instance_type=instance_type,
    instance_count=1,
    disable_profiler=True,
    debugger_hook_config=False,
    role=SAGEMAKER_EXECUTION_ROLE,
    sagemaker_session=sagemaker_session,
)

training_step = TrainingStep(
    name="train-model",
    step_args=tf_estimator.fit(
        inputs={
            "train": TrainingInput(
                s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv",
                # input_mode=
            ),
            "validation": TrainingInput(
                s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
                content_type="text/csv",
            ),
            # NOTE: I could not use "preprocessing-pipeline" as the input channel name because when using Sagemaker built-in
            # Tensorflow Estimator, the training script could not recognise the "SM_CHANNEL_PREPROCESSING-PIPELINE" env. var.
            # even though the env var existed after looking the CloudWatch logs. But this works using a custom estimator.
            # So for "Tensorflow" Estimator, we settled on using "preprocessing_pipeline" as the input channel name.
            "preprocessing_pipeline": TrainingInput(
                s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs[
                    "preprocessing-pipeline"
                ].S3Output.S3Uri,
                content_type="application/tar+gzip",
            ),
        },
    ),
    cache_config=cache_config,
)


# Create the pipeline
pipeline = Pipeline(
    name="e2e-ml-pipeline",
    parameters=[dataset_location],
    steps=[
        preprocessing_step,
        training_step,
    ],
    sagemaker_session=sagemaker_session,
    pipeline_definition_config=pipeline_definition_config,
)

if __name__ == "__main__":
    # # Note: sagemaker.get_execution_role does not work outside sagemaker
    # role = sagemaker.get_execution_role()
    pipeline.upsert(role_arn=SAGEMAKER_EXECUTION_ROLE)
    pipeline.start()

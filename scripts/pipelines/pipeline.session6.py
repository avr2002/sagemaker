"""
Setup Sagemaker Preprocessing Job Pipeline.

https://github.com/aws/amazon-sagemaker-examples/tree/main
"""

# A processing step requires a processor, a Python script that defines
# the processing code, outputs for processing, and job arguments.
import os
from pathlib import Path

import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.parameter import IntegerParameter
from sagemaker.processing import FrameworkProcessor, ProcessingInput, ProcessingOutput
from sagemaker.tuner import HyperparameterTuner
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import LocalPipelineSession, PipelineSession
from sagemaker.workflow.pipeline_definition_config import PipelineDefinitionConfig
from sagemaker.workflow.steps import CacheConfig, ProcessingStep, TuningStep

from penguins.consts import BUCKET, LOCAL_MODE, S3_LOCATION, SAGEMAKER_EXECUTION_ROLE, SAGEMAKER_PROCESSING_DIR
from penguins.utils.docker import build_and_push_docker_image

env_vars = {
    "COMET_API_KEY": os.getenv("COMET_API_KEY", ""),
    "COMET_PROJECT_NAME": os.getenv("COMET_PROJECT_NAME", ""),
    "LOCAL_MODE": LOCAL_MODE,
    "S3_BUCKET_NAME": BUCKET,
    "SAGEMAKER_EXECUTION_ROLE": SAGEMAKER_EXECUTION_ROLE,
}


THIS_DIR = Path(__file__).parent

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
cache_config = CacheConfig(enable_caching=False, expire_after="T3H")  # 3 hours


# ref: https://github.com/aws/amazon-sagemaker-examples/blob/main/sagemaker_processing/scikit_learn_data_processing_and_model_evaluation/scikit_learn_data_processing_and_model_evaluation.ipynb
est_cls = sagemaker.sklearn.estimator.SKLearn
# ref: https://docs.aws.amazon.com/sagemaker/latest/dg/sklearn.html
framework_version_str = "1.2-1"  # Sagemaker available Scikit-learn version

framework_processor = FrameworkProcessor(
    base_job_name="data-preprocessing",
    role=SAGEMAKER_EXECUTION_ROLE,
    instance_count=1,
    instance_type=instance_type,
    estimator_cls=est_cls,
    framework_version=framework_version_str,
    sagemaker_session=sagemaker_session,
    env=env_vars,
    # Control where preprocessing code artifacts are stored
    code_location=f"{S3_LOCATION}/preprocessing/",
    # py_version="py3",
)

preprocessing_step = ProcessingStep(
    name="data-preprocessing",
    step_args=framework_processor.run(
        code="src/penguins/preprocessor.py",
        # source_dir="src/penguins",     # Check the doc-string to know more about this parameter
        # While installing the local package (via pip install .) in requirements.txt, we need to pass the pyproject.toml and README.md files
        # Right now, we are not installing the local package because of dependency conflicts
        # dependencies=["src/penguins", "requirements.txt", "pyproject.toml", "README.md"],
        dependencies=["src/penguins", "requirements.txt"],
        inputs=[
            ProcessingInput(
                source=dataset_location,
                destination=(SAGEMAKER_PROCESSING_DIR / "input").as_posix(),
            ),
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

# Automatically build and push the Docker image
# image_uri = build_and_push_docker_image(
#     # Create a custom container with keras with jax as the backend
#     repository_name="custom-keras-jax-training-container",
#     dockerfile_fpath=THIS_DIR / "containers/training/Dockerfile.keras.jax",
# )
image_uri = build_and_push_docker_image(
    # Create a custom container with tensorflow keras -- We will serve this model using TF Serving
    repository_name="custom-keras-tf-serving-container",
    dockerfile_fpath=THIS_DIR / "containers/training/Dockerfile.tf.keras",
)

custom_estimator = Estimator(
    base_job_name="custom-training-job",
    image_uri=image_uri,
    entry_point="src/penguins/train.py",
    # container_entry_point=["python", "/opt/ml/code/train.py"],
    # source_dir="src/penguins",  # Check the doc-string to know more about this parameter
    dependencies=["src/penguins"],
    # SageMaker will pass these hyperparameters as arguments
    # to the entry point of the training script.
    hyperparameters={"epochs": 50, "batch_size": 32},
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
    # Control where the model gets saved in S3
    output_path=f"{S3_LOCATION}/training/",
    # Control where training code artifacts are stored
    code_location=f"{S3_LOCATION}/training/",
    instance_type=instance_type,
    instance_count=1,
    disable_profiler=True,
    debugger_hook_config=False,
    role=SAGEMAKER_EXECUTION_ROLE,
    sagemaker_session=sagemaker_session,
)

# https://sagemaker.readthedocs.io/en/stable/api/training/tuner.html#
# https://github.com/aws-samples/amazon-sagemaker-hyperparameter-tuning-portfolio-optimization/blob/master/hyperparameter-tuning-portfolio-optimization/Using_Amazon_Sagemaker_To_Optimize_Portfolio_Value.ipynb
# NOTE: Step type Tuning is not supported in local mode.
tuner = HyperparameterTuner(
    base_tuning_job_name="hpo-job",
    estimator=custom_estimator,
    objective_metric_name="val_accuracy",
    objective_type="Maximize",  # This value can be either 'Minimize' or 'Maximize'
    hyperparameter_ranges={
        "epochs": IntegerParameter(10, 50),
        # other types of parameters can be -
        # CategoricalParameter - A class for representing hyperparameters that have a discrete list of possible values
        # ContinuousParameter - A class for representing hyperparameters that have a continuous range of possible values.
    },
    # Metric Definitions: https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-define-metrics-variables.html
    metric_definitions=[
        {"Name": "val_accuracy", "Regex": "val_accuracy: ([0-9\\.]+)"},
    ],
    # hyperparameter tuning strategies available in Amazon SageMaker AI
    # https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-how-it-works.html
    strategy="Bayesian",  # Options: 'Bayesian', 'Random', 'Hyperband', 'Grid' (default: 'Bayesian')
    max_jobs=3,  # Max number of training jobs to launch
    max_parallel_jobs=3,  # Max number of training jobs to run in parallel
)

# While configuring the hyperparameter ranges, make sure training script accepts the hyperparameters
# as arguments --epochs, --batch_size, --learning_rate
# hyperparameter_ranges = {
#     "epochs": IntegerParameter(10, 50),
#     "batch_size": IntegerParameter(16, 64),
#     "learning_rate": ContinuousParameter(1e-5, 1e-2),
# }

# In Sagemaker:
# "SM_HPS" environment variable contains a JSON encoded dictionary with the user provided hyperparameters
# We can access individual hyperparameters using env var: "SM_HP_{hyperparameter_name}". E.g: SM_HP_LEARNING-RATE=0.0001, SM_HP_BATCH-SIZE=10000

tuning_step = TuningStep(
    name="tune-model",
    step_args=tuner.fit(
        inputs={
            "train": TrainingInput(
                s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv",
                # input_mode=  # Explore different input modes
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

# Access Model artifacts of tuning step for model evaluation step
# https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_model_building_pipeline.html#tuningstep

# What's the difference b/w these two?
# Get the best model trained, top_k=0
# model_assets = tuning_step.get_top_model_s3_uri(top_k=0, s3_bucket=BUCKET)

# https://sagemaker.readthedocs.io/en/stable/api/training/tuner.html#sagemaker.tuner.HyperparameterTuner.best_estimator
# best_estimator = tuner.best_estimator(best_training_job=tuner.best_training_job())


# Create the pipeline
pipeline = Pipeline(
    name="e2e-ml-pipeline",
    parameters=[dataset_location],
    steps=[
        preprocessing_step,
        tuning_step,
    ],
    sagemaker_session=sagemaker_session,
    pipeline_definition_config=pipeline_definition_config,
)

if __name__ == "__main__":
    # # Note: sagemaker.get_execution_role does not work outside sagemaker environment
    # role = sagemaker.get_execution_role()
    pipeline.upsert(role_arn=SAGEMAKER_EXECUTION_ROLE)
    pipeline.start()

"""
Sagemaker Pipeline with dynamic model registration step and pipeline failure notification step.

ref:
https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_model_building_pipeline.html#
https://github.com/aws/amazon-sagemaker-examples/tree/main
"""

# A processing step requires a processor, a Python script that defines
# the processing code, outputs for processing, and job arguments.
import os
from pathlib import Path
from typing import Any

import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import FrameworkProcessor, ProcessingInput, ProcessingOutput
from sagemaker.tensorflow.model import TensorFlowModel
from sagemaker.tensorflow.processing import TensorFlowProcessor
from sagemaker.workflow.callback_step import CallbackStep
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import LocalPipelineSession, PipelineSession
from sagemaker.workflow.pipeline_definition_config import PipelineDefinitionConfig
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import CacheConfig, ProcessingStep, TrainingStep

from penguins.consts import (
    BUCKET,
    LOCAL_MODE,
    S3_LOCATION,
    SAGEMAKER_EXECUTION_ROLE,
    SAGEMAKER_PROCESSING_DIR,
)
from penguins.utils.docker import build_and_push_docker_image, build_docker_image

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
    if os.getenv("LOCAL_MODE", None)
    else PipelineSession(default_bucket=BUCKET)
)
# https://docs.aws.amazon.com/sagemaker/latest/dg/notebooks-available-instance-types.html
# locally instance_type can also be "local_gpu"
instance_type = "local" if os.getenv("LOCAL_MODE", None) else "ml.m5.2xlarge"  # "ml.m5.xlarge"

# define a parameter for the input data
dataset_location = ParameterString(name="dataset-location", default_value=f"{S3_LOCATION}/data/")

# Setup Cache for the pipeline step
# ref: https://sagemaker.readthedocs.io/en/stable/workflows/pipelines/sagemaker.workflow.pipelines.html#sagemaker.workflow.steps.CacheConfig

# If caching is enabled, the pipeline attempts to find a previous execution of a Step that was called with the same arguments.
# Step caching only considers successful execution. If a successful previous execution is found, the pipeline propagates the values
# from the previous execution rather than recomputing the Step. When multiple successful executions exist within the timeout period,
# it uses the result for the most recent successful execution.
cache_config = CacheConfig(
    enable_caching=False,  # Enable or disable caching
    expire_after="T3H",  # expiration time in ISO8601 duration string format - https://en.wikipedia.org/wiki/ISO_8601#Durations
)
# p30d: 30 days
# P4DT12H: 4 days and 12 hours
# T12H: 12 hours


#######################
### Processing Step ###
#######################


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
    name="preprocess-data",
    display_name="Preprocess Data",
    step_args=framework_processor.run(
        # job_name="data-preprocessing",
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
                # ClientError: An error occurred (ValidationException) when calling the CreatePipeline operation:
                # Unable to parse pipeline definition. Model Validation failed:
                # Value 'FastFile' for 'ProcessingS3InputMode' failed to satisfy enum value set: [Pipe, File]
                s3_input_mode="File",  # "Pipe", "File".
                # "FastFile" is present in docs but not supported in pipeline
                # -----------------
                # ^^NOTE: Using "Pipe" mode, I got this error raise ValueError(f"No CSV files found in {input_directory.as_posix()}")
                # ^^^NOTE: Using "Pipe" Mode usually gives file not found error. Where would Pipe mode be appropriate?
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name=output_name,
                source=f"{SAGEMAKER_PROCESSING_DIR}/{output_name}",
                destination=f"{S3_LOCATION}/preprocessing/{output_name}/",  # Added a trailing '/' because I got an error using "FastFile" mode in Training Step
                s3_upload_mode=(
                    "EndOfJob" if os.getenv("LOCAL_MODE", None) else "Continuous"
                ),  # "Continuous" or "EndOfJob"
                # ^^^NOTE: RuntimeError: UploadMode: Continuous is not currently supported in Local Mode.
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
# ^^^NOTE: I got this Error in the training Step when using "FastFile" input mode in training.
#
# ClientError: ClientError: Data download failed:Failed to download data.
# For FastFile mode, partial S3 prefix is not supported. S3 prefix has to end with a full file or folder.
# Add a trailing '/' to the S3 prefix if it ends with a folder name that is also a prefix for other folders.
# Current S3 prefix: [penguins/preprocessing/train], found S3 object: [penguins/preprocessing/train-baseline/train-baseline.csv].

# ^^^Question: Will adding a trailing '/' work with other modes?

#####################
### Training Step ###
#####################

# Automatically build and push the Docker image
# image_uri = build_and_push_docker_image(
#     # Create a custom container with keras with jax as the backend
#     repository_name="custom-keras-jax-training-container",
#     dockerfile_fpath=THIS_DIR / "containers/training/Dockerfile.keras.jax",
# )
image_uri = build_and_push_docker_image(
    # Create a custom container with tensorflow keras -- We will serve this model using TF Serving
    repository_name="custom-tf-keras-training-container",
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
training_step = TrainingStep(
    name="train-model",
    display_name="Train Model",
    step_args=custom_estimator.fit(
        inputs={
            "train": TrainingInput(
                s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv",
                input_mode="FastFile",  # None, Pipe, File or FastFile
            ),
            "validation": TrainingInput(
                s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
                content_type="text/csv",
                input_mode="FastFile",  # None, Pipe, File or FastFile
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
                input_mode="File",  # None, Pipe, File or FastFile
            ),
        },
    ),
    cache_config=cache_config,
)
# ^^^NOTE: I got this error while mixing input modes in Training Job
# ClientError: Failed to invoke sagemaker:CreateTrainingJob.
# Error Details: Cannot use both 'Pipe' and 'FastFile' TrainingInputMode on the same Training Job.


#######################
### Evaluation Step ###
#######################

# Sagemaker does not have a built-in model evaluation job, so will use the "Processing Job" to create one.

# Access Model artifacts of training step for model evaluation step --
model_assets = training_step.properties.ModelArtifacts.S3ModelArtifacts

# Use property files to store information from the output of a processing step. This is particularly useful
# when analyzing the results of a processing step to decide how a conditional step should be executed.
# ref: https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-propertyfile.html
evaluation_report = PropertyFile(
    name="evaluation-report",
    output_name="evaluation",
    path="evaluation.json",  # The file name in the output directory
)
# When you create your ProcessingStep instance, add the property_files parameter to list all of the parameter
# files that the Amazon SageMaker Pipelines service must index. This saves the property file for later use.


if os.getenv("LOCAL_MODE", None):
    # image_uri = "sagemaker-tf-training-toolkit-arm64:latest"
    eval_step_image_uri = build_docker_image(
        repository_name="sagemaker-tf-training-toolkit-arm64",
        dockerfile_fpath=THIS_DIR / "containers/Dockerfile",
        tag="latest",
    )
else:
    eval_step_image_uri = None

# The model group name used for Model Registry
MODEL_PACKAGE_GROUP_NAME = "basic-penguins-model-group"

evaluation_processor = TensorFlowProcessor(
    base_job_name="model-evaluation-job",
    code_location=f"{S3_LOCATION}/evaluation/",
    framework_version="2.12.0",
    py_version="py310",
    image_uri=eval_step_image_uri,
    instance_count=1,
    instance_type=instance_type,
    env={
        **env_vars,
        "MODEL_PACKAGE_GROUP_NAME": MODEL_PACKAGE_GROUP_NAME,
    },
    role=SAGEMAKER_EXECUTION_ROLE,
    sagemaker_session=sagemaker_session,
)

evaluation_step = ProcessingStep(
    name="evaluate-model",
    display_name="Evaluate Model",
    step_args=evaluation_processor.run(
        code="src/penguins/evaluate.py",
        dependencies=["src/penguins", "requirements.txt"],
        # For Model Evaluation, we need the "test" dataset from "Pre-processing Step" and the trained model
        # from "Model Training/Model Tuning Step".
        inputs=[
            ProcessingInput(
                source=preprocessing_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                destination=str(SAGEMAKER_PROCESSING_DIR / "test"),
                s3_input_mode="File",  # "Pipe", "File"
                # "FastFile" is present in docs but not supported in pipeline
            ),
            ProcessingInput(
                source=model_assets,
                destination=str(SAGEMAKER_PROCESSING_DIR / "model"),
                s3_input_mode="File",  # "Pipe", "File"
                # "FastFile" is present in docs but not supported in pipeline
            ),
            # ^^^NOTE: Using "Pipe" Mode usually gives file not found error. Where would Pipe mode be appropriate?
            # FileNotFoundError: [Errno 2] No such file or directory: '/opt/ml/processing/model/model.tar.gz'
        ],
        outputs=[
            # The output is the evaluation report that we generated in the evaluation script
            ProcessingOutput(
                output_name="evaluation",  # The output name must match the "PropertyFile" output name
                source=str(SAGEMAKER_PROCESSING_DIR / "evaluation"),
                destination=f"{S3_LOCATION}/evaluation",
                s3_upload_mode="EndOfJob",  # "Continuous" if not os.getenv("LOCAL_MODE", None) else "EndOfJob"
                # "Continuous" or "EndOfJob"
                # ^^^NOTE: RuntimeError: UploadMode: Continuous is not currently supported in Local Mode.
            )
        ],
    ),
    # When you create your ProcessingStep instance, add the property_files parameter to list all of the parameter
    # files that the Amazon SageMaker Pipelines service must index. This saves the property file for later use.
    property_files=[evaluation_report],
    cache_config=cache_config,
)


##########################
### Model Registration ###
##########################

# ref: Sagemaker: https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html
# Neptune: https://neptune.ai/blog/ml-model-registry
# MLflow: https://mlflow.org/docs/latest/ml/model-registry/
# ZenML: https://docs.zenml.io/stacks/stack-components/model-registries


# To register a model in SageMaker Model Registry, we create a "SageMaker Model" object which is nothing but
# an abstraction over the trained model assets/artifacts, along with model performance metrics (accuracy, precision, etc.).
# And we use the "ModelStep" in the pipeline to register the model in the SageMaker Model Registry.
# This "Model" object can later be used for model serving.


# Create a Sagemaker Model

# from sagemaker.model import Model
# Model(
#     image_uri=custom_estimator.image_uri,
#     model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,  # model_assets
#     source_dir="src/penguins",
#     code_location=f"{S3_LOCATION}/training/",
#     entry_point="src/penguins/train.py",
#     role=SAGEMAKER_EXECUTION_ROLE,
#     sagemaker_session=sagemaker_session,
# )
tf_model = TensorFlowModel(
    model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,  # model_assets
    framework_version="2.12.0",
    role=SAGEMAKER_EXECUTION_ROLE,
    sagemaker_session=sagemaker_session,
)

# Create Model Metrics Object
model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri=Join(
            on="/",
            values=[
                # this gives the S3 dir. path, so we need to join it with the file name
                evaluation_step.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri,
                "evaluation.json",
            ],
        ),
        content_type="application/json",
    )
)
# ^^^NOTE: We cannot simply do string concatenation with S3Uri_dir_path + "/evaluation.json"
# Because the `evaluation_step.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri`
# is not a regular Python String - it's a SageMaker pipeline property that gets resolved at pipeline execution time.

# So, Pipeline functions like "Join, JsonGet" are used to assign values to properties that are not available until pipeline execution time.
# ref: https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_model_building_pipeline.html#pipeline-functions

# Register the model using model step
# MODEL_PACKAGE_GROUP_NAME = "basic-penguins-model-group"
register_model_step = ModelStep(
    name="register-model",
    display_name="Register Model",
    step_args=tf_model.register(
        model_package_group_name=MODEL_PACKAGE_GROUP_NAME,
        model_metrics=model_metrics,
        drift_check_baselines=None,
        approval_status="Approved",  # Literal["Approved", "Rejected", "PendingManualApproval"]
        content_types=["text/csv"],  # The content type of the model input
        response_types=["application/json"],  # The type of prediction the model sends back
        transform_instances=[instance_type],  # The instance type(s) that can be used for batch transform jobs
        inference_instances=[instance_type],  # The instance type(s) that can be used for real-time inference
        domain="MACHINE_LEARNING",  # Literal["COMPUTER_VISION", "NATURAL_LANGUAGE_PROCESSING", "MACHINE_LEARNING"]
        task="CLASSIFICATION",  # Literal["OBJECT_DETECTION", "TEXT_GENERATION", "IMAGE_SEGMENTATION", "CLASSIFICATION", "REGRESSION", "OTHER"]
        framework="TENSORFLOW",
        framework_version="2.12.0",
    ),
)
# ^^^NOTE: No Local Mode for Model Registration Step
# ClientError: An error occurred (ValidationException) when calling the start_pipeline_execution operation: Step type RegisterModel is not supported in
# local mode.


#########################################
### Conditional Pipeline Registration ###
#########################################

# Register the model only if the model performance is better than the latest registered model version
# """
# if current_model_accuracy >= latest_registered_model_accuracy:
#     register_model_step
# else:
#     fail_step
# """

# Create a FailStep that will fail the pipeline if the model performance is not better than baseline
fail_step = FailStep(
    name="fail-step",
    display_name="Fail Pipeline",
    description="Fail the pipeline if model accuracy is not better than the latest registered model.",
    error_message=Join(
        on=" ",
        values=[
            "Execution failed because model's accuracy",
            JsonGet(
                step_name=evaluation_step.name,
                property_file=evaluation_report,
                json_path="metrics.accuracy.value",
            ),
            "is not better than baseline accuracy",
            JsonGet(
                step_name=evaluation_step.name,
                property_file=evaluation_report,
                json_path="metrics.baseline_accuracy.value",
            ),
        ],
    ),
)

# Define the condition to check whether the current model accuracy is better than baseline
# ref: https://sagemaker.readthedocs.io/en/stable/workflows/pipelines/sagemaker.workflow.pipelines.html#conditions
condition = ConditionGreaterThanOrEqualTo(
    left=JsonGet(  # type: ignore[arg-type]
        step_name=evaluation_step.name,
        property_file=evaluation_report,
        json_path="metrics.accuracy.value",  # The path to the accuracy value in the evaluation report
    ),
    # right=0.99,
    right=JsonGet(  # type: ignore[arg-type]
        step_name=evaluation_step.name,
        property_file=evaluation_report,
        json_path="metrics.baseline_accuracy.value",  # The path to the baseline accuracy value in the evaluation report
    ),
)


# ref: CallbackStep: https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps-types.html
# https://aws.amazon.com/blogs/machine-learning/extend-amazon-sagemaker-pipelines-to-include-custom-steps-using-callback-steps/
# callback-step-for-batch-transform: https://github.com/aws-samples/sagemaker-pipelines-callback-step-for-batch-transform/tree/main

# Setting up a CallbackStep to notify the pipeline failures and send email notifications
failure_notification_inputs: dict[str, Any] = {
    key: JsonGet(
        step_name=evaluation_step.name,
        property_file=evaluation_report,
        json_path=f"metrics.{key}.value",
    )
    for key in ["accuracy", "baseline_accuracy", "precision", "recall"]
}
failure_notification_inputs["model_package_group"] = MODEL_PACKAGE_GROUP_NAME
failure_notification_inputs["step_name"] = evaluation_step.name

notify_pipeline_failure_step = CallbackStep(
    name="notify-pipeline-failure",
    display_name="Notify Pipeline Failure",
    sqs_queue_url=os.environ["SQS_QUEUE_URL"],
    inputs=failure_notification_inputs,
    outputs=[],
)

# Set the Condition Step
condition_step = ConditionStep(
    name="check-model-performance",
    display_name="Check Model Performance vs Baseline",
    conditions=[condition],
    if_steps=[register_model_step],
    else_steps=[notify_pipeline_failure_step, fail_step],
)


################
### Pipeline ###
################


# WARNING:sagemaker.workflow.utilities:Popping out 'TrainingJobName' from the pipeline definition
# by default since it will be overridden at pipeline execution time.
# Please utilize the PipelineDefinitionConfig to persist this field in the pipeline definition if desired.
pipeline_definition_config = PipelineDefinitionConfig(use_custom_job_prefix=True)

# Create the pipeline
# A pipeline is a series of interconnected steps that is defined by a JSON pipeline definition.
# JSON Schema: https://aws-sagemaker-mlops.github.io/sagemaker-model-building-pipeline-definition-JSON-schema/
pipeline = Pipeline(
    name="e2e-ml-pipeline",
    parameters=[dataset_location],
    steps=[
        preprocessing_step,
        training_step,
        evaluation_step,
        # get_baseline_step,
        condition_step,
    ],
    sagemaker_session=sagemaker_session,
    pipeline_definition_config=pipeline_definition_config,
)

if __name__ == "__main__":
    # # Note: sagemaker.get_execution_role does not work outside sagemaker environment
    # role = sagemaker.get_execution_role()

    # Submit the pipeline definition to the Pipelines service to create a pipeline if it doesn't exist, or update the pipeline if it does.
    # The role passed in is used by Pipelines to create all of the jobs defined in the steps.
    pipeline.upsert(
        role_arn=SAGEMAKER_EXECUTION_ROLE,
        description="ML Pipeline to train model on Palmer Penguins dataset.",
    )

    # Starts the pipeline execution.
    pipeline.start(
        # # You can start the pipeline execution with specific parameters
        # parameters={"accuracy-threshold": 0.99},
        # execution_display_name="penguins-pipeline-execution",
        # execution_description="Executing Pipeline, overiding the default accuracy threshold to reach the fail step.",
    )

    # execution = pipeline.start()
    # More info: https://docs.aws.amazon.com/sagemaker/latest/dg/run-pipeline.html

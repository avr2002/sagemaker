import os
import platform
from pathlib import Path

import boto3
from dotenv import load_dotenv

# from sagemaker.local import LocalSession
from sagemaker.processing import (  # FrameworkProcessor,
    ProcessingInput,
    ProcessingOutput,
    Processor,
)

# from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import (
    LocalPipelineSession,
    PipelineSession,
)
from sagemaker.workflow.pipeline_definition_config import PipelineDefinitionConfig
from sagemaker.workflow.steps import (
    CacheConfig,
    ProcessingStep,
)

from penguins.consts import (
    BUCKET,
    LOCAL_MODE,
    S3_LOCATION,
    SAGEMAKER_EXECUTION_ROLE,
    SAGEMAKER_PROCESSING_DIR,
)

# from sagemaker.sklearn.estimator import SKLearn


load_dotenv()

# Constants
THIS_DIR: Path = Path(__file__).resolve().parent
SOURCE_DIR: str = str(THIS_DIR / "src")

# Get the architecture of the local machine
architecture = platform.machine()
IS_ARM64_ARCHITECTURE = architecture == "arm64"


def get_ecr_image_uri(repository_name, region="ap-south-1"):
    client = boto3.client("ecr", region_name=region)
    response = client.describe_repositories()
    for repo in response["repositories"]:
        if repo["repositoryName"] == repository_name:
            return repo["repositoryUri"]
    raise ValueError(f"Repository {repository_name} not found in {region}.")


REPOSITORY_NAME = "processing-job"
ECR_IMAGE_URI = get_ecr_image_uri(REPOSITORY_NAME)

# Sagemaker Session Setup
pipeline_session: LocalPipelineSession | PipelineSession = (
    LocalPipelineSession(default_bucket=BUCKET) if LOCAL_MODE else PipelineSession(default_bucket=BUCKET)
)

if isinstance(pipeline_session, LocalPipelineSession):
    pipeline_session.config = {"local": {"local_code": True}}

config = {
    "session": pipeline_session,
    "instance_type": "local" if LOCAL_MODE else "ml.m5.xlarge",
    "image": "sagemaker-tensorflow-toolkit-local" if LOCAL_MODE and IS_ARM64_ARCHITECTURE else None,
    "framework_version": "2.12",
    "py_version": "py310",
}

# Parameters
pipeline_definition_config = PipelineDefinitionConfig(use_custom_job_prefix=True)
dataset_location = ParameterString(name="dataset_location", default_value=f"{S3_LOCATION}/data")


processor: Processor = Processor(
    image_uri=ECR_IMAGE_URI,
    role=SAGEMAKER_EXECUTION_ROLE,
    instance_count=1,
    instance_type=config["instance_type"],
    sagemaker_session=config["session"],
    env={
        "S3_BUCKET_NAME": os.environ["S3_BUCKET_NAME"],
        "SAGEMAKER_EXECUTION_ROLE": os.environ["SAGEMAKER_EXECUTION_ROLE"],
        "LOCAL_MODE": os.environ["LOCAL_MODE"],
        "AWS_REGION": "ap-south-1",
        "AWS_DEFAULT_REGION": "ap-south-1",
    },
)


# Enable caching in Processing Step
cache_config: CacheConfig = CacheConfig(enable_caching=True, expire_after="15d")

# Define the processing step
preprocessing_step = ProcessingStep(
    name="preprocess-data",
    step_args=processor.run(
        inputs=[ProcessingInput(source=dataset_location, destination=f"{str(SAGEMAKER_PROCESSING_DIR)}/input")],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source=f"{str(SAGEMAKER_PROCESSING_DIR)}/train",
                destination=f"{S3_LOCATION}/preprocessing/train",
            ),
            ProcessingOutput(
                output_name="validation",
                source=f"{str(SAGEMAKER_PROCESSING_DIR)}/validation",
                destination=f"{S3_LOCATION}/preprocessing/validation",
            ),
            ProcessingOutput(
                output_name="test",
                source=f"{str(SAGEMAKER_PROCESSING_DIR)}/test",
                destination=f"{S3_LOCATION}/preprocessing/test",
            ),
            ProcessingOutput(
                output_name="model",
                source=f"{str(SAGEMAKER_PROCESSING_DIR)}/model",
                destination=f"{S3_LOCATION}/preprocessing/model",
            ),
            ProcessingOutput(
                output_name="train-baseline",
                source=f"{str(SAGEMAKER_PROCESSING_DIR)}/train-baseline",
                destination=f"{S3_LOCATION}/preprocessing/train-baseline",
            ),
            ProcessingOutput(
                output_name="test-baseline",
                source=f"{str(SAGEMAKER_PROCESSING_DIR)}/test-baseline",
                destination=f"{S3_LOCATION}/preprocessing/test-baseline",
            ),
        ],
    ),
    cache_config=cache_config,
)

# Pipeline Definition
pipeline: Pipeline = Pipeline(
    name="penguins-preprocessing-pipeline",
    parameters=[dataset_location],
    steps=[preprocessing_step],
    pipeline_definition_config=pipeline_definition_config,
    sagemaker_session=config["session"],
)

pipeline.upsert(role_arn=SAGEMAKER_EXECUTION_ROLE)


if __name__ == "__main__":
    pipeline.start()

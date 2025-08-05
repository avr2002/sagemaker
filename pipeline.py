"""Setup Sagemaker Preprocessing Job Pipeline."""

# A processing step requires a processor, a Python script that defines
# the processing code, outputs for processing, and job arguments.
import os

import sagemaker
from dotenv import load_dotenv
from sagemaker.processing import FrameworkProcessor, ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import LocalPipelineSession

# from sagemaker.workflow.pipeline_definition_config import PipelineDefinitionConfig
from sagemaker.workflow.steps import CacheConfig, ProcessingStep

from penguins.consts import BUCKET, S3_LOCATION, SAGEMAKER_EXECUTION_ROLE, SAGEMAKER_PROCESSING_DIR, SRC_DIR

# Load environment variables from .env file
load_dotenv()
env_vars = {
    # "COMET_API_KEY": os.getenv("COMET_API_KEY", ""),
    # "COMET_PROJECT_NAME": os.getenv("COMET_PROJECT_NAME", ""),
    "S3_BUCKET_NAME": os.getenv("S3_BUCKET_NAME", ""),
    "SAGEMAKER_EXECUTION_ROLE": os.getenv("SAGEMAKER_EXECUTION_ROLE", ""),
    "LOCAL_MODE": os.getenv("LOCAL_MODE", ""),
}

# Create a local Sagemaker session
sagemaker_session = LocalPipelineSession(default_bucket=BUCKET)

# WARNING:sagemaker.workflow.utilities:Popping out 'ProcessingJobName' from the pipeline definition
# by default since it will be overridden at pipeline execution time.
# Please utilize the PipelineDefinitionConfig to persist this field in the pipeline definition if desired.
# pipeline_definition_config = PipelineDefinitionConfig(use_custom_job_prefix=True)

# define a parameter for the input data
dataset_location = ParameterString(name="dataset-location", default_value=f"{S3_LOCATION}/data/")

# Setup Cache for the pipeline step
cache_config = CacheConfig(enable_caching=True, expire_after="15d")


est_cls = sagemaker.sklearn.estimator.SKLearn
framework_version_str = "1.2-1"

framework_processor = FrameworkProcessor(
    base_job_name="data-preprocessing",
    role=SAGEMAKER_EXECUTION_ROLE,
    instance_count=1,
    instance_type="local",
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
        code="preprocessor.py",
        source_dir="src/penguins",
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
            for output_name in ["train", "validation", "test", "model", "train-baseline", "test-baseline"]
        ],
    ),
    cache_config=cache_config,
)

# Create the pipeline
pipeline = Pipeline(
    name="data-preprocessing-pipeline",
    parameters=[dataset_location],
    steps=[preprocessing_step],
    sagemaker_session=sagemaker_session,
)

if __name__ == "__main__":
    # # Note: sagemaker.get_execution_role does not work outside sagemaker
    # role = sagemaker.get_execution_role()
    pipeline.upsert(role_arn=SAGEMAKER_EXECUTION_ROLE)
    pipeline.start()

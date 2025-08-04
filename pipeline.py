"""Setup Sagemaker Preprocessing Job Pipeline."""

# A processing step requires a processor, a Python script that defines
# the processing code, outputs for processing, and job arguments.

from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import LocalPipelineSession

# from sagemaker.workflow.pipeline_definition_config import PipelineDefinitionConfig
from sagemaker.workflow.steps import CacheConfig, ProcessingStep

from penguins.consts import BUCKET, S3_LOCATION, SAGEMAKER_EXECUTION_ROLE, SAGEMAKER_PROCESSING_DIR, SRC_DIR

sagemaker_processing_dir = SAGEMAKER_PROCESSING_DIR.as_posix()

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

sklearn_processor = SKLearnProcessor(
    base_job_name="data-preprocessing",
    framework_version="1.2-1",
    role=SAGEMAKER_EXECUTION_ROLE,
    instance_type="local",
    instance_count=1,
    sagemaker_session=sagemaker_session,
)

preprocessing_step = ProcessingStep(
    name="data-preprocessing",
    step_args=sklearn_processor.run(
        code=(SRC_DIR / "penguins" / "preprocessor.py").as_posix(),
        inputs=[
            ProcessingInput(source=dataset_location, destination=sagemaker_processing_dir),
        ],
        outputs=[
            ProcessingOutput(
                output_name=output_name,
                source=f"{sagemaker_processing_dir}/{output_name}",
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
pipeline.upsert(role_arn=SAGEMAKER_EXECUTION_ROLE)

if __name__ == "__main__":
    pipeline.start()

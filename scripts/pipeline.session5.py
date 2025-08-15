"""
Setup Sagemaker Preprocessing Job Pipeline.

Sagemaker Model Training: https://docs.aws.amazon.com/sagemaker/latest/dg/train-model.html

https://github.com/aws/amazon-sagemaker-examples/tree/main
"""

# A processing step requires a processor, a Python script that defines
# the processing code, outputs for processing, and job arguments.
import os
from pathlib import Path

import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.processing import FrameworkProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import LocalPipelineSession, PipelineSession

# from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.pipeline_definition_config import PipelineDefinitionConfig
from sagemaker.workflow.steps import CacheConfig, ProcessingStep, TrainingStep

from penguins.consts import BUCKET, LOCAL_MODE, S3_LOCATION, SAGEMAKER_EXECUTION_ROLE, SAGEMAKER_PROCESSING_DIR
from penguins.utils import build_and_push_docker_image

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

# Create a custom container with keras and jax
# Automatically build and push the Docker image
repository_name = "custom-keras-training-container"
image_uri = build_and_push_docker_image(
    repository_name=repository_name,
    dockerfile_fpath=THIS_DIR / "containers/training",
)

custom_estimator = Estimator(
    base_job_name="custom-training-job",
    image_uri=image_uri,
    entry_point="train.py",
    # container_entry_point=["python", "/opt/ml/code/train.py"],
    source_dir="src/penguins",
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

# For Sagemaker Training, the inputs to the pipeline can be access with the environment variables prefixed with "SM_CHANNEL_"
# The env var, "SM_CHANNELS", contains the list of all the inputs
# https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_Channel.html
# https://pypi.org/project/sagemaker-containers/#important-environment-variables

# NOTE: Sagemaker does not automatically converts dashes to underscores in channel names
# So, if a channel name is "preprocessing-pipeline", then the corresponding env var for it will be "SM_CHANNEL_PREPROCESSING-PIPELINE"
training_step = TrainingStep(
    name="train-model",
    step_args=custom_estimator.fit(
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

# # Access model artifacts for model evaluation step
# model_assets = training_step.properties.ModelArtifacts.S3ModelArtifacts

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
    # # Note: sagemaker.get_execution_role does not work outside sagemaker environment
    # role = sagemaker.get_execution_role()
    pipeline.upsert(role_arn=SAGEMAKER_EXECUTION_ROLE)
    pipeline.start()


# Using "code_location" argument in the steps, where Sagemaker uploads the code artifacts
# But in pre-processing job has a sagemaker managed shellscript called "runproc.sh" which is not uploaded in
# the specified code location.

# """
# s3://ml-school-bucket-6721/
# ├── penguins/
# │   ├── data/
# │   ├── preprocessing/
# │   │   ├── e2e-ml-pipeline/
# │   │   │   └── code/
# │   │   │       └── 4472b7991005fc3aaeacc2e77f357aea29c1ee50a9624ff03f8a6884f4eb4015/
# │   │   │           └── sourcedir.tar.gz
# │   ├── training/
# │   │   ├── custom-training-job-5w5r8jajcdeo-nX2fnIHvDU/
# │   │   │   └── output/
# │   │   │       └── model.tar.gz
# │   │   └── e2e-ml-pipeline/
# │   │       └── code/
# │   │           └── e5ad0140453be771469b5c4acc7df805d47dc3e82435e16aac4d4e153e67442c/
# │   │               └── sourcedir.tar.gz
# ├── e2e-ml-pipeline/
# │   └── code/
# │       └── f036ce328456a216b40166796ad96c92cd30a9c2ef4bda333085138e7d9b80c0/
# │           └── runproc.sh
# """

# """
# runproc.sh is a script that is automatically generated by Amazon SageMaker to manage the execution of your processing job.
# It is responsible for setting up the processing environment, running your custom processing code, and handling the upload of output data to Amazon S3.

# The runproc.sh script is stored in a separate location (s3://<bucket>/e2e-ml-pipeline/code/<hash>/runproc.sh)
# because it is a system-generated file that is not part of your custom processing code.
# The code_location parameter you specified is intended for your own custom processing scripts and dependencies, not for system-generated files.

# There is no direct way to store the runproc.sh file in your specified code_location.
# The location of this file is determined by the SageMaker Processing service and is not configurable.
# However, you can still access the contents of the runproc.sh file if needed, as it is stored in the same S3
# location as your custom processing code.

# To ensure that all artifacts generated by a step of your pipeline are in one place, you can use the
# ProcessingOutput parameter to specify the S3 location where you want your processing job outputs to be stored.
# This will ensure that all output data from your processing job, including any intermediate files or artifacts,
# are uploaded to the same S3 location.
# """


"""
# Questions I asked AWS Support:

Q1. What exactly is runproc.sh and what role it plays in the processing job?

    - In Amazon SageMaker Processing jobs, runproc.sh file is an internal shell script that SageMaker automatically generates and uses as an entry point for your processing job. 
    - When you use a Docker image for a SageMaker Processing job, runproc.sh is the script that SageMaker executes within your container to initiate your processing logic.
    - It performs several critical functions including setting up the runtime environment, managing input and output paths, handling logging configuration, and orchestrating the execution of your actual processing script.
    - From the below pipeline definition, you can observe that the runproc.sh file serves as a entry point.

    ```
        "AppSpecification": {
          "ImageUri": "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
          "ContainerEntrypoint": [
            "/bin/bash",
            "/opt/ml/processing/input/entrypoint/runproc.sh"
          ]
        },

    ...

          {
            "InputName": "entrypoint",
            "AppManaged": false,
            "S3Input": {
              "S3Uri": "s3://ml-school-bucket-6721/e2e-ml-pipeline/code/f036ce328456a216b40166796ad96c92cd30a9c2ef4bda333085138e7d9b80c0/runproc.sh",
              "LocalPath": "/opt/ml/processing/input/entrypoint",
              "S3DataType": "S3Prefix",
              "S3InputMode": "File",
              "S3DataDistributionType": "FullyReplicated",
              "S3CompressionType": "None"
            }
          }
    ```

Q2. Why is it not stored under the code_location you have provided?

    - The runproc.sh file is not stored in your specified code_location because it is considered part of SageMaker's internal processing infrastructure. 
    - SageMaker maintains a separation between user-provided code artifacts (which respect your code_location parameter) and system-generated files that are required for the proper functioning of the processing job and to avoid conflicts and to manage those files independently (e.g., lifecycle, cleanup).

Q3. Is there a way where runproc.sh can be stored in your specified code_location?

    -  Unfortunately, there is currently no direct way to force runproc.sh to be stored in your specified code_location. This is by design, as SageMaker needs to maintain control over system-generated files to ensure proper job execution.
    - According to the document[1], this is how the path is generated and managed by SageMaker source code.

    ```
            s3_uri = s3.s3_path_join(
                "s3://",
                self.sagemaker_session.default_bucket(),
                self.sagemaker_session.default_bucket_prefix,
                _pipeline_config.pipeline_name,
                "code",
                runproc_file_hash,
                "runproc.sh",
            )
    ```

References:
=========
[1] https://github.com/aws/sagemaker-python-sdk/blob/9bfe85abe338375ea870b8bda6635d04e8d7fc4b/src/sagemaker/processing.py#L2032C1-L2040C14 
[2] https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateProcessingJob.html  
[3] https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_AppSpecification.html#sagemaker-Type-AppSpecification-ContainerEntrypoint
"""

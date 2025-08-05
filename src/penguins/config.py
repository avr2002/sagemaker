# import platform

# from sagemaker.workflow.pipeline_context import (
#     LocalPipelineSession,
#     PipelineSession,
# )

# from penguins.consts import (
#     BUCKET,
#     LOCAL_MODE,
# )

# # Get the architecture of the local machine
# architecture = platform.machine()
# IS_ARM64_ARCHITECTURE = architecture == "arm64"

# # Sagemaker Session Setup
# pipeline_session: LocalPipelineSession | PipelineSession = (
#     LocalPipelineSession(default_bucket=BUCKET) if LOCAL_MODE else PipelineSession(default_bucket=BUCKET)
# )
# config: dict = {
#     "session": pipeline_session,
#     "instance_type": "local" if LOCAL_MODE else "ml.m5.xlarge",
#     "image": "sagemaker-tensorflow-toolkit-local" if LOCAL_MODE and IS_ARM64_ARCHITECTURE else None,
#     "framework_version": "2.12",
#     "py_version": "py310",
# }

import os
from pathlib import Path
from typing import Optional

# Define the paths to the directories
THIS_DIR: Path = Path(__file__).resolve().parent
SRC_DIR: Path = THIS_DIR.parent
ROOT_DIR: Path = SRC_DIR.parent
DATA_DIR: Path = ROOT_DIR / "data"

# SAGEMAKER BASE DIRECTORIES
SAGEMAKER_BASE_DIR: Path = Path("/opt/ml")

# Define sagemaker processing directories
SAGEMAKER_PROCESSING_DIR: Path = SAGEMAKER_BASE_DIR / "processing"

# SageMaker training directories
SAGEMAKER_MODEL_DIR: Path = SAGEMAKER_BASE_DIR / "model"


# Define the environment variables
BUCKET: str = os.environ["S3_BUCKET_NAME"]
S3_LOCATION: str = "s3://{bucket}/penguins".format(bucket=BUCKET)
LOCAL_MODE: str | None = os.getenv("LOCAL_MODE", None)

SAGEMAKER_EXECUTION_ROLE: str = os.environ["SAGEMAKER_EXECUTION_ROLE"]
COMET_API_KEY: Optional[str] = os.environ.get("COMET_API_KEY", None)
COMET_PROJECT_NAME: Optional[str] = os.environ.get("COMET_PROJECT_NAME", None)

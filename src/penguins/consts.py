import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

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
SAGEMAKER_EXECUTION_ROLE: str = os.environ["SAGEMAKER_EXECUTION_ROLE"]
LOCAL_MODE: bool = os.getenv("LOCAL_MODE", "true").lower() == "true"  # Convert to boolean, Defaults to true
COMET_API_KEY: str | None = os.environ.get("COMET_API_KEY", None)
COMET_PROJECT_NAME: str | None = os.environ.get("COMET_PROJECT_NAME", None)

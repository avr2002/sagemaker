import base64
import os
import subprocess
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError

try:
    from mypy_boto3_ecr.client import ECRClient
    from mypy_boto3_ecr.type_defs import GetAuthorizationTokenResponseTypeDef
except ImportError:
    ...


THIS_DIR = Path(__file__).parent


# """
# Docker manifest v2 Error:
# ref:
# - https://repost.aws/questions/QUYwYL0SKaTQWvQMyqCSBvdQ/manifest-error-when-staging-ml-model
# - https://forums.docker.com/t/force-image-to-use-manifest-media-type-docker-v2-schema-2-instead-of-oci/144974/3
# - https://docs.aws.amazon.com/AmazonECR/latest/userguide/image-manifest-formats.html
# """


def build_and_push_docker_image(
    repository_name: str,
    dockerfile_fpath: Path | str,
    tag: str = "latest",
    force_rebuild: bool = False,
) -> str:
    """
    Build and push Docker image to ECR if it doesn't exist or if running locally.

    Args:
        repository_name (str): Name of the ECR repository
        dockerfile_fpath (Path | str): Path to the directory containing the Dockerfile
        tag (str): Docker image tag, defaults to "latest"
        force_rebuild (bool): Force rebuild even if image exists in ECR

    Returns:
        str: Image URI for the built image
    """
    dockerfile_fpath = Path(dockerfile_fpath)

    if os.getenv("LOCAL_MODE"):
        # For local mode, just build the image locally
        build_docker_image(repository_name, dockerfile_fpath, tag=tag)
        return f"{repository_name}:{tag}"

    # For SageMaker mode, build and push to ECR
    aws_account_id = boto3.client("sts").get_caller_identity()["Account"]
    aws_region = boto3.Session().region_name or "us-east-1"
    ecr_uri = f"{aws_account_id}.dkr.ecr.{aws_region}.amazonaws.com/{repository_name}:{tag}"

    ecr_client: "ECRClient" = boto3.client("ecr")

    # Create ECR repository if it doesn't exist
    _create_ecr_repo_if_not_exists(repository_name, ecr_client)

    # Check if image exists in ECR (skip if force_rebuild is True)
    if not force_rebuild and _check_if_image_exists(repository_name, tag, ecr_client):
        return ecr_uri

    try:
        # Build the Docker image locally first
        build_docker_image(repository_name, dockerfile_fpath, tag=tag)

        # Login to ECR
        # https://boto3.amazonaws.com/v1/documentation/api/1.29.2/reference/services/ecr/client/get_authorization_token.html
        _login_to_ecr(ecr_client)

        # # Tag and push the image
        # subprocess.run(["docker", "tag", f"{repository_name}:{tag}", ecr_uri], check=True)

        # push_env = os.environ.copy()
        # push_env["DOCKER_CLI_EXPERIMENTAL"] = "enabled"
        # subprocess.run(["docker", "push", ecr_uri, "--disable-content-trust"], check=True, env=push_env)

        # Push with Docker v2 manifest format (required by SageMaker)

        # Use buildx to push with Docker v2 manifest format (SageMaker compatible)
        # This avoids OCI manifest issues that SageMaker doesn't support
        buildx_command = [
            "docker",
            "buildx",
            "build",
            "--platform",
            "linux/amd64",
            "--provenance=false",  # Disable provenance attestations
            "--sbom=false",  # Disable SBOM attestations
            "--output",
            f"type=image,name={ecr_uri},push=true,oci-mediatypes=false",
            "--file",
            str(dockerfile_fpath),
            f"{THIS_DIR.parent.parent.parent.as_posix()}",
        ]

        subprocess.run(buildx_command, check=True, cwd=dockerfile_fpath.parent, env=os.environ.copy())

        print(f"Successfully built and pushed Docker image to {ecr_uri}")
        return ecr_uri

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to build and push Docker image: {e}")


def build_docker_image(repository_name: str, dockerfile_fpath: Path, tag: str = "latest") -> str:
    """Build Docker image locally.

    :param repository_name: Name of the Docker repository
    :param dockerfile_fpath: Path to the Dockerfile
    :param tag: Tag for the Docker image

    :return: local image URI, "<repository_name>:<tag>"
    """
    # Check if dockerfile_fpath is path to a dockerfile
    if not dockerfile_fpath.is_file():
        raise ValueError(f"Invalid Dockerfile path: {dockerfile_fpath=}")

    # Set environment to force Docker v2 manifest format
    build_env = os.environ.copy()

    if os.getenv("LOCAL_MODE"):
        command = f"docker build --tag {repository_name}:{tag} --file {str(dockerfile_fpath)} {THIS_DIR.parent.parent.parent.as_posix()}"
    else:
        build_env["DOCKER_BUILDKIT"] = "0"  # Disable BuildKit to use legacy builder
        build_env["DOCKER_DEFAULT_PLATFORM"] = "linux/amd64"  # Set default platform
        # For SageMaker environment build the image for amd64 architecture
        command = f"docker build --platform linux/amd64 --tag {repository_name}:{tag} --file {str(dockerfile_fpath)} {THIS_DIR.parent.parent.parent.as_posix()}"
    try:
        subprocess.run(
            command.split(),
            check=True,
            cwd=dockerfile_fpath.parent,  # Set working directory to dockerfile directory
            env=build_env,  # Use environment with DOCKER_BUILDKIT=0
            # capture_output=True,
            # text=True,
        )
        print(f"Successfully built Docker image {repository_name}:{tag}")
        return f"{repository_name}:{tag}"
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to build Docker image: {e}")


def _check_if_image_exists(repository_name: str, tag: str, ecr_client: Optional["ECRClient"] = None) -> bool:
    """Check if a Docker image exists in ECR."""
    ecr_client: "ECRClient" = ecr_client or boto3.client("ecr")
    try:
        ecr_client.describe_images(repositoryName=repository_name, imageIds=[{"imageTag": tag}])
        print(f"Docker image {repository_name}:{tag} already exists in ECR")
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "ImageNotFoundException":
            print(f"Docker image {repository_name}:{tag} not found in ECR")
            return False
        else:
            raise


def _create_ecr_repo_if_not_exists(repository_name: str, ecr_client: Optional["ECRClient"] = None) -> None:
    ecr_client: "ECRClient" = ecr_client or boto3.client("ecr")
    try:
        ecr_client.create_repository(repositoryName=repository_name)
        print(f"Created ECR Repository: {repository_name=}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "RepositoryAlreadyExistsException":
            print(f"ECR Repository already exists: {repository_name=}")
        else:
            raise


def _login_to_ecr(ecr_client: Optional["ECRClient"] = None) -> None:
    ecr_client: "ECRClient" = ecr_client or boto3.client("ecr")

    login_response: "GetAuthorizationTokenResponseTypeDef" = ecr_client.get_authorization_token()
    auth_token = login_response["authorizationData"][0]["authorizationToken"]
    proxy_endpoint = login_response["authorizationData"][0]["proxyEndpoint"]

    decoded_token = base64.b64decode(auth_token).decode("utf-8")
    username, password = decoded_token.split(":", 1)

    # docker expects host without scheme: e.g. https://<acct>.dkr.ecr.<region>.amazonaws.com
    registry = proxy_endpoint.replace("https://", "")

    subprocess.run(
        ["docker", "login", "--username", username, "--password-stdin", registry],
        input=password,
        check=True,
        # capture_output=True,
        text=True,
    )


# if __name__ == "__main__":
#     THIS_DIR = Path(__file__).parent

#     repository_name = "custom-keras-training-container"
#     image_uri = build_and_push_docker_image(
#         repository_name=repository_name,
#         dockerfile_fpath=THIS_DIR / "../../containers/training",
#         force_rebuild=True,
#     )

#!/bin/bash

set -e

AWS_PROFILE="ml.school"
AWS_REGION="ap-south-1"
S3_BUCKET_NAME="ml-school-bucket-avr"

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Data file path
DATA_FILE_PATH="$THIS_DIR/data/penguins.csv" # Path to the data file on your local machine
FILE_NAME="$(basename "$DATA_FILE_PATH")"
S3_FILE_PATH="$FILE_NAME"       # "penguins/data/$FILE_NAME"     # Path to the data file in the S3 bucket


##########################
# --- Task Functions --- #
##########################


# Install Python dependencies into the currently activated venv
function install {
    # if uv is not installed, then update the commands as `python -m pip install --upgrade pip`
	# uv run pip install --upgrade pip
	# uv run pip install --editable "$THIS_DIR"

	python -m pip install --upgrade pip
	python -m pip install --editable "$THIS_DIR/[dev]"

    # If using uv and you are getting pyyaml compilation error during installation then follow the below thread:
    # https://github.com/astral-sh/uv/issues/1455
    # This error doesn't occur when using the normal python command.
    # The error happend with python version 3.10.x [NOT SURE]
}


# Activate your AWS_PROFILE for the project
function set-local-aws-env-vars {
    export AWS_PROFILE
    export AWS_REGION
    # export $S3_BUCKET_NAME
}

# Setup AWS resources for the project
function setup-aws {
    set -e

    # Remove old AWS credentials
    remove-old-aws-credentials

    # Append new AWS credentials to .env
    aws configure export-credentials --profile "$AWS_PROFILE" --format env >> .env

    # Set local AWS environment variables
    set-local-aws-env-vars

    # Create S3 bucket and upload data
    setup-aws-s3

    # Set the SageMaker Execution Role in the .env file
    setup-sagemaker-execution-role
}


function setup-aws-s3 {
    # Create S3 bucket if it doesn't exist
    create-s3-bucket

    # Upload data to S3 bucket
    upload-data-to-s3 "$DATA_FILE_PATH"

    # Set the S3 bucket name in the .env file
    sed -i '/^S3_BUCKET_NAME=/d' .env  # Remove old S3_BUCKET_NAME entry
    echo "S3_BUCKET_NAME=$S3_BUCKET_NAME" >> .env

    echo "S3 bucket name has been set in .env file."
}

# Check if a file exists and upload data to S3 bucket
function upload-data-to-s3 { ## Check if a file exists and upload it to the S3 bucket
    # Ensure a file path argument is provided
    if [[ -z "$1" ]]; then
        echo "Error: No file path provided."
        echo "Usage: upload-data-to-s3 <file_path>"
        return 1
    fi

    # File to upload
    local file_path="$1"

    # Extract the file name (base name) from the path
    local file_name=$(basename "$file_path")
    local s3_file_path="$S3_FILE_PATH"

    # Check if the file exists
    if [[ ! -f "$file_path" ]]; then
        echo "Error: File $file_path does not exist."
        return 1
    fi

    # Upload the file to S3
    echo "Uploading file $file_path to S3 bucket $S3_BUCKET_NAME..."
    aws s3 cp "$file_path" "s3://$S3_BUCKET_NAME/$s3_file_path" > /dev/null

    if [[ $? -eq 0 ]]; then
        echo "File $file_name successfully uploaded to S3 bucket $S3_BUCKET_NAME at $s3_file_path."
    else
        echo "Error: Failed to upload $file_path to S3 bucket $S3_BUCKET_NAME."
        return 1
    fi
}


# Create S3 bucket if it doesn't exist
function create-s3-bucket {
    if ! aws s3api head-bucket --bucket $S3_BUCKET_NAME > /dev/null 2>&1; then
        echo "Creating S3 bucket $S3_BUCKET_NAME in $AWS_REGION..."

        aws s3api create-bucket \
            --bucket $S3_BUCKET_NAME \
            --create-bucket-configuration LocationConstraint="$AWS_REGION" > /dev/null || true

        echo "S3 bucket $S3_BUCKET_NAME created."
    fi
}


# Set the SageMaker Execution Role in the .env file

# To grab the correct domain_id & user_profile_name, modify the index of
# the UserProfiles array in the jq command if you have multiple user profiles
function setup-sagemaker-execution-role {
    domain_id=$(aws sagemaker list-user-profiles --profile=$AWS_PROFILE \
               | jq -r ".UserProfiles[0].DomainId")

    user_profile_name=$(aws sagemaker list-user-profiles --profile=$AWS_PROFILE \
                        | jq -r ".UserProfiles[0].UserProfileName")

    execution_role=$(aws sagemaker describe-user-profile --profile=$AWS_PROFILE \
                     --domain-id $domain_id --user-profile-name $user_profile_name \
                     | jq -r ".UserSettings.ExecutionRole")

    # Set the execution role in the .env file
    sed -i '/^SAGEMAKER_EXECUTION_ROLE=/d' .env  # Remove old ROLE entry
    echo "SAGEMAKER_EXECUTION_ROLE=$execution_role" >> .env

    echo "SageMaker Execution Role has been set in .env file."
}

# Function to remove old AWS credentials from the .env file
function remove-old-aws-credentials {
    # Delete any lines that export AWS credentials
    sed -i '/^export AWS_ACCESS_KEY_ID/d' .env
    sed -i '/^export AWS_SECRET_ACCESS_KEY/d' .env
    sed -i '/^export AWS_SESSION_TOKEN/d' .env
    sed -i '/^export AWS_CREDENTIAL_EXPIRATION/d' .env

    # Ensure there is a new line at the end of the .env file if it's missing, so that the new credentials are on a new line
    if [ "$(tail -c 1 .env)" != "" ]; then
        echo "" >> .env
    fi
}


# run linting, formatting, and other static code quality tools
function lint {
	pre-commit run --all-files
}

# same as `lint` but with any special considerations for CI
function lint:ci {
	# We skip no-commit-to-branch since that blocks commits to `main`.
	# All merged PRs are commits to `main` so this must be disabled.
	SKIP=no-commit-to-branch pre-commit run --all-files
}

# remove all files generated by tests, builds, or operating this codebase
function clean {
	rm -rf dist build coverage.xml test-reports
	find . \
	  -type d \
	  \( \
		-name "*cache*" \
		-o -name "*.dist-info" \
		-o -name "*.egg-info" \
		-o -name "*htmlcov" \
	  \) \
	  -not -path "*env*/*" \
	  -exec rm -r {} + || true

	find . \
	  -type f \
	  -name "*.pyc" \
	  -not -path "*env/*" \
	  -exec rm {} +
}


# print all functions in this file
function help {
    echo "$0 <task> <args>"
    echo "Tasks:"
    compgen -A function | cat -n
}


TIMEFORMAT="Task completed in %3lR"
time ${@:-help}

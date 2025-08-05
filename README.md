# Machine Learning School

This repository contains my *modified* source code of the [Machine Learning School](https://www.ml.school).


## Setup

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) before running.

- If you are using AWS SSO, you can activate your profile by running the following command:

    ```bash
    # AWS_PROFILE=sandbox
    aws configure sso --profile sandbox
    # OR
    aws sso login --profile sandbox
    ```

- Setup Infra
    ```bash
    # Bootstrap the CDK environment
    ./run cdk-bootstrap

    # Deploy the CDK stack
    # This will create the S3 bucket and SageMaker domain
    ./run cdk-deploy
    ```

<!-- - Create a `.env` file and add
  - `COMET_API_KEY`, `COMET_PROJECT_NAME`
  - `S3_BUCKET_NAME`, `SAGEMAKER_EXECUTION_ROLE` -->



## Contributing

If you find any problems with the code or have any ideas on improving it, please open an issue and share your recommendations.

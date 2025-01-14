# Machine Learning School

This repository contains my *modified* source code of the [Machine Learning School](https://www.ml.school).


## Setup

- If you are using AWS SSO, you can activate your profile by running the following command:

    ```bash
    # AWS_PROFILE=ml.school
    aws configure sso --profile ml.school
    # OR
    aws sso login --profile ml.school
    ```

- If you're new to SSO, then please follow the documentation to set it up and configure your profile.


- If you are using permanent credentials, then make sure you have your secrets configured in `~/.aws/credentials`:

    ```bash
    [ml.school]
    aws_access_key_id = YOUR_ACCESS_KEY
    aws_secret_access_key = YOUR_SECRET_KEY
    ```

- Finally in `run.sh` script, make sure to set the `AWS_PROFILE` & `AWS_REGION` variables to their correct values.

- Create a `.env` file and add your `COMET_API_KEY` and `COMET_PROJECT_NAME` to it.

- Create your Python virtual environment and activate it:

    ```bash
    pyenv install 3.10  # Install Python 3.10
    pyenv local 3.10
    python -m venv .venv
    source .venv/bin/activate
    ```

- Run:
  - If `uv` is not installed, then either [install it](https://docs.astral.sh/uv/getting-started/installation/) OR if you don't want to use it then update the `install` function in `run.sh` accordingly.
  - `make install` to install the required dependencies.
  - `make setup-aws` to setup the all the AWS resources.

- You should be ready to work with the notebook!


## Running the code

For now run the notebook.


## TODO:

- [x] Simplified the codebase, moved everything to toml file.

- [x] Generated a uv lock file

- [] There is some problem installing using `uv`, [conflict with `pyyaml` version](https://github.com/astral-sh/uv/issues/1455) and `protobuf` version for `tensorflow` & `sagemaker` aka DEPENDENCY CONFLICT/HELL.

- [] Make the code modular and get out of notebooks!!


## Contributing

If you find any problems with the code or have any ideas on improving it, please open an issue and share your recommendations.
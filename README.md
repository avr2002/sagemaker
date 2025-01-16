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
  - Setup Sagemaker domain prior to running the script.
    In the AWS Console, `Sagemaker > Admin configurations > Domains > Create domain > Opt for Quick Setup > Click Set up`

  - If you have multiple Sagemaker domains configured make sure to update the `setup-sagemaker-execution-role` function in `run.sh`,
    so that it grabs the correct domain_id and user_profile_name.

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


- Update the permission associated to the IAM Role of your Sagemaker domain.
  - Grab your Execution role ID in your Sagemaker Domain UI OR if have followed the steps so far it should be in your `.env` file, and should look something like this `AmazonSageMaker-ExecutionRole-xxxxxxxx`

  - Go your IAM Roles, find that role, update the permissions for the same `AmazonSageMaker-ExecutionPolicy-xxxxxxx` with below JSON. Also update the Trust policy in the same Role UI.

    - <details>
      <summary>Update the Permissions for that Execution role</summary>

      ```
      {
          "Version": "2012-10-17",
          "Statement": [
              {
                  "Sid": "IAM0",
                  "Effect": "Allow",
                  "Action": [
                      "iam:CreateServiceLinkedRole"
                  ],
                  "Resource": "*",
                  "Condition": {
                      "StringEquals": {
                          "iam:AWSServiceName": [
                              "autoscaling.amazonaws.com",
                              "ec2scheduled.amazonaws.com",
                              "elasticloadbalancing.amazonaws.com",
                              "spot.amazonaws.com",
                              "spotfleet.amazonaws.com",
                              "transitgateway.amazonaws.com"
                          ]
                      }
                  }
              },
              {
                  "Sid": "IAM1",
                  "Effect": "Allow",
                  "Action": [
                      "iam:CreateRole",
                      "iam:DeleteRole",
                      "iam:PassRole",
                      "iam:AttachRolePolicy",
                      "iam:DetachRolePolicy",
                      "iam:CreatePolicy"
                  ],
                  "Resource": "*"
              },
              {
                  "Sid": "Lambda",
                  "Effect": "Allow",
                  "Action": [
                      "lambda:CreateFunction",
                      "lambda:DeleteFunction",
                      "lambda:InvokeFunctionUrl",
                      "lambda:InvokeFunction",
                      "lambda:UpdateFunctionCode",
                      "lambda:InvokeAsync",
                      "lambda:AddPermission",
                      "lambda:RemovePermission"
                  ],
                  "Resource": "*"
              },
              {
                  "Sid": "SageMaker",
                  "Effect": "Allow",
                  "Action": [
                      "sagemaker:UpdateDomain",
                      "sagemaker:UpdateUserProfile"
                  ],
                  "Resource": "*"
              },
              {
                  "Sid": "CloudWatch",
                  "Effect": "Allow",
                  "Action": [
                      "cloudwatch:PutMetricData",
                      "cloudwatch:GetMetricData",
                      "cloudwatch:DescribeAlarmsForMetric",
                      "logs:CreateLogStream",
                      "logs:PutLogEvents",
                      "logs:CreateLogGroup",
                      "logs:DescribeLogStreams"
                  ],
                  "Resource": "*"
              },
              {
                  "Sid": "ECR",
                  "Effect": "Allow",
                  "Action": [
                      "ecr:GetAuthorizationToken",
                      "ecr:BatchCheckLayerAvailability",
                      "ecr:GetDownloadUrlForLayer",
                      "ecr:BatchGetImage"
                  ],
                  "Resource": "*"
              },
              {
                  "Sid": "S3",
                  "Effect": "Allow",
                  "Action": [
                      "s3:CreateBucket",
                      "s3:ListBucket",
                      "s3:GetBucketLocation",
                      "s3:PutObject",
                      "s3:GetObject",
                      "s3:DeleteObject"
                  ],
                  "Resource": "arn:aws:s3:::*"
              },
              {
                  "Sid": "EventBridge",
                  "Effect": "Allow",
                  "Action": [
                      "events:PutRule",
                      "events:PutTargets"
                  ],
                  "Resource": "*"
              }
          ]
      }
      ```

    </details>

    - <details>
          <summary>Update the Trust Policy</summary>

          ```
          {
              "Version": "2012-10-17",
              "Statement": [
                  {
                      "Effect": "Allow",
                      "Principal": {
                          "Service": [
                              "sagemaker.amazonaws.com",
                              "events.amazonaws.com"
                          ]
                      },
                      "Action": "sts:AssumeRole"
                  }
              ]
          }
          ```

      </details>

- If you're confused then watch this [video](https://youtu.be/153BboqWh-U)!

- You should be ready to work with the notebook!


## Running the code

For now run the notebook.


## Changelog

- [x] Simplified the codebase, moved everything to `pyproject.toml` file and created a `src` directory structure
- [x] Automated project onbarding like setting up `S3` and `Sagemaker` using bash script
- [x] Generated a uv lock file

- [ ] There is some problem installing using `uv`, [conflict with `pyyaml` version](https://github.com/astral-sh/uv/issues/1455) and `protobuf` version for `tensorflow` & `sagemaker` aka DEPENDENCY CONFLICT/HELL.

- [ ] Make the code modular and get out of notebooks!!


## Contributing

If you find any problems with the code or have any ideas on improving it, please open an issue and share your recommendations.

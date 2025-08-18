# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "aws-cdk-lib>=2.201.0",
#     "constructs>=10.4.2",
# ]
# ///


# Official AWS Samples Repo Guide - https://github.com/aws-samples/aws-cdk-sagemaker-studio/tree/main
# https://aws.amazon.com/blogs/machine-learning/automate-amazon-sagemaker-studio-setup-using-aws-cdk/

import os

import aws_cdk as cdk
from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as _lambda
from aws_cdk import aws_lambda_event_sources as lambda_event_sources
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_s3_deployment as s3_deployment
from aws_cdk import aws_sagemaker as sagemaker
from aws_cdk import aws_sns as sns
from aws_cdk import aws_sns_subscriptions as sns_subscriptions
from aws_cdk import aws_sqs as sqs
from constructs import Construct

#################
# --- Stack --- #
#################

# VPC_NAME = "default"
VPC_NAME = "local-virginia"


class SagemakerMLStack(Stack):
    """Stack for SageMaker ML infrastructure."""

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # 1. Create S3 Bucket for ML data and artifacts
        self.ml_bucket = s3.Bucket(
            self,
            "MLSchoolBucket",
            # bucket_name=f"ml-school-bucket-{construct_id.lower()}",
            bucket_name=os.environ["S3_BUCKET_NAME"],
            versioned=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            # lifecycle_rules=[
            #     s3.LifecycleRule(
            #         id="DeleteIncompleteMultipartUploads",
            #         abort_incomplete_multipart_upload_after=Duration.days(7),
            #         enabled=True,
            #     ),
            #     s3.LifecycleRule(
            #         id="TransitionToIA",
            #         transitions=[
            #             s3.Transition(
            #                 storage_class=s3.StorageClass.INFREQUENT_ACCESS,
            #                 transition_after=Duration.days(30),
            #             )
            #         ],
            #         enabled=True,
            #     ),
            # ],
        )

        # 2. Create ECR Repository for Docker images
        # self.ecr_repository = ecr.Repository(
        #     self,
        #     "ProcessingJobRepository",
        #     repository_name="processing-job",
        #     image_scan_on_push=True,
        #     lifecycle_rules=[
        #         ecr.LifecycleRule(
        #             description="Keep only 10 most recent images",
        #             max_image_count=10,
        #         )
        #     ],
        #     removal_policy=RemovalPolicy.DESTROY,
        # )

        # 3. Create SageMaker Execution Role with comprehensive permissions
        self.sagemaker_execution_role = iam.Role(
            self,
            "SageMakerExecutionRole",
            role_name=f"SageMaker-ExecutionRole-{construct_id}",
            assumed_by=iam.CompositePrincipal(
                iam.ServicePrincipal("sagemaker.amazonaws.com"),
                iam.ServicePrincipal("events.amazonaws.com"),
            ),
            description="SageMaker execution role for ML School project",
        )

        # Add comprehensive policies to SageMaker execution role
        self._add_sagemaker_policies()

        # 3a. Create notification infrastructure
        self.notification_queue = self._create_notification_queue()
        self.notification_topic = self._create_notification_topic()
        self.notification_lambda = self._create_notification_lambda()

        # 4. Create SageMaker Domain (optional - can be created manually)
        # Uncomment if you want CDK to manage the SageMaker domain
        self.sagemaker_domain = self._create_sagemaker_domain()

        # 5. Create default user profile for the SageMaker domain
        self.default_user_profile = self._create_default_user_profile()

        # 6. Deploy sample data to S3 (if data file exists)
        self._deploy_sample_data()

        # 7. Create outputs
        self._create_outputs()

    def _create_notification_queue(self) -> sqs.Queue:
        """Create SQS queue for pipeline failure notifications."""
        dlq = sqs.Queue(
            self,
            "NotificationDLQ",
            queue_name="mlschool-pipeline-notifications-dlq",
            retention_period=Duration.days(7),
        )

        return sqs.Queue(
            self,
            "NotificationQueue",
            queue_name="mlschool-pipeline-notifications",
            visibility_timeout=Duration.minutes(5),
            receive_message_wait_time=Duration.seconds(20),
            dead_letter_queue=sqs.DeadLetterQueue(
                max_receive_count=3,
                queue=dlq,
            ),
        )

    def _create_notification_topic(self) -> sns.Topic:
        """Create SNS topic for team notifications."""
        topic = sns.Topic(
            self,
            "NotificationTopic",
            topic_name="mlschool-pipeline-alerts",
            display_name="ML School Pipeline Alerts",
        )

        # NOTE: Adding Subscriptions via CDK did not work for some reason.
        # I would confirm my subscription, and then the next moment I would get another email saying I got unsubscribed

        # # Add email subscriptions: https://docs.aws.amazon.com/cdk/api/v2/python/aws_cdk.aws_sns_subscriptions/README.html
        # for email in ["amit.raj@pattern.com", "eric.riddoch@pattern.com"]:
        #     topic.add_subscription(topic_subscription=sns_subscriptions.EmailSubscription(email_address=email))

        return topic

    def _create_notification_lambda(self) -> _lambda.Function:
        """Create Lambda function to process notifications."""
        # Create Lambda execution role
        lambda_role = iam.Role(
            self,
            "NotificationLambdaRole",
            role_name="MLSchool-Lambda-Sagemaker-Notification-Role",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole"),
                # Add SQS permissions -- saw this managed policy in AWS Console
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaSQSQueueExecutionRole"),
            ],
        )

        # Add permissions for SNS, and SageMaker
        lambda_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["sns:Publish"],
                resources=[self.notification_topic.topic_arn],
            )
        )

        # For the Callback Step to complete in Sagemaker, Lambda needs to tell Sagemaker notifying with Step Success/Failure
        lambda_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "sagemaker:SendPipelineExecutionStepSuccess",
                    "sagemaker:SendPipelineExecutionStepFailure",
                ],
                resources=["*"],
            )
        )

        # Create Lambda function
        notification_lambda = _lambda.Function(
            self,
            "NotificationLambda",
            function_name=f"ml-pipeline-notification-handler-{self.stack_name.lower()}",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler="pipeline_failure_notification_handler.lambda_handler",
            code=_lambda.Code.from_asset("./scripts/lambdas/"),
            role=lambda_role,
            timeout=Duration.minutes(5),
            environment={
                "SNS_TOPIC_ARN": self.notification_topic.topic_arn,
            },
        )

        # Add SQS trigger to Lambda
        notification_lambda.add_event_source(
            lambda_event_sources.SqsEventSource(
                self.notification_queue,
                batch_size=1,
                report_batch_item_failures=True,
            )
        )

        return notification_lambda

    def _add_sagemaker_policies(self) -> None:
        """Add comprehensive IAM policies to SageMaker execution role."""

        # Add AWS managed SageMaker policy
        self.sagemaker_execution_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess")
        )

        # Custom policy for additional permissions
        custom_policy = iam.PolicyDocument(
            statements=[
                # IAM permissions
                iam.PolicyStatement(
                    sid="IAM0",
                    effect=iam.Effect.ALLOW,
                    actions=["iam:CreateServiceLinkedRole"],
                    resources=["*"],
                    conditions={
                        "StringEquals": {
                            "iam:AWSServiceName": [
                                "autoscaling.amazonaws.com",
                                "ec2scheduled.amazonaws.com",
                                "elasticloadbalancing.amazonaws.com",
                                "spot.amazonaws.com",
                                "spotfleet.amazonaws.com",
                                "transitgateway.amazonaws.com",
                            ]
                        }
                    },
                ),
                iam.PolicyStatement(
                    sid="IAM1",
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "iam:CreateRole",
                        "iam:DeleteRole",
                        "iam:PassRole",
                        "iam:AttachRolePolicy",
                        "iam:DetachRolePolicy",
                        "iam:CreatePolicy",
                    ],
                    resources=["*"],
                ),
                # Lambda permissions
                iam.PolicyStatement(
                    sid="Lambda",
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "lambda:CreateFunction",
                        "lambda:DeleteFunction",
                        "lambda:InvokeFunctionUrl",
                        "lambda:InvokeFunction",
                        "lambda:UpdateFunctionCode",
                        "lambda:InvokeAsync",
                        "lambda:AddPermission",
                        "lambda:RemovePermission",
                    ],
                    resources=["*"],
                ),
                # SageMaker permissions
                iam.PolicyStatement(
                    sid="SageMaker",
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "sagemaker:UpdateDomain",
                        "sagemaker:UpdateUserProfile",
                    ],
                    resources=["*"],
                ),
                # CloudWatch permissions
                iam.PolicyStatement(
                    sid="CloudWatch",
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "cloudwatch:PutMetricData",
                        "cloudwatch:GetMetricData",
                        "cloudwatch:DescribeAlarmsForMetric",
                        "logs:CreateLogStream",
                        "logs:PutLogEvents",
                        "logs:CreateLogGroup",
                        "logs:DescribeLogStreams",
                    ],
                    resources=["*"],
                ),
                # ECR permissions
                iam.PolicyStatement(
                    sid="ECR",
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "ecr:GetAuthorizationToken",
                        "ecr:BatchCheckLayerAvailability",
                        "ecr:GetDownloadUrlForLayer",
                        "ecr:BatchGetImage",
                    ],
                    resources=["*"],
                ),
                # S3 permissions
                iam.PolicyStatement(
                    sid="S3",
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "s3:CreateBucket",
                        "s3:ListBucket",
                        "s3:GetBucketLocation",
                        "s3:PutObject",
                        "s3:GetObject",
                        "s3:DeleteObject",
                    ],
                    resources=["arn:aws:s3:::*"],
                ),
                # EventBridge permissions
                iam.PolicyStatement(
                    sid="EventBridge",
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "events:PutRule",
                        "events:PutTargets",
                    ],
                    resources=["*"],
                ),
                # SQS permissions for CallbackStep
                iam.PolicyStatement(
                    sid="SQS",
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "sqs:SendMessage",
                        "sqs:GetQueueAttributes",
                        "sqs:GetQueueUrl",
                    ],
                    resources=["*"],
                    # [self.notification_queue.queue_arn] if hasattr(self, "notification_queue") else ["*"],
                ),
            ]
        )

        self.sagemaker_execution_role.attach_inline_policy(
            iam.Policy(
                self,
                "SageMakerAICustomPolicy",
                document=custom_policy,
            )
        )

    def _create_sagemaker_domain(self) -> sagemaker.CfnDomain:
        """Create SageMaker Domain."""
        # https://github.com/aws-samples/aws-cdk-sagemaker-studio/blob/main/sagemakerStudioCDK/sagemaker_studio_stack.py

        # Get default VPC (you might want to specify a custom VPC)
        from aws_cdk import aws_ec2 as ec2

        # vpc = ec2.Vpc.from_lookup(self, "DefaultVPC", is_default=True)
        if VPC_NAME.lower() == "default":
            vpc = ec2.Vpc.from_lookup(self, "VpcLookup", is_default=True)
        else:
            vpc = ec2.Vpc.from_lookup(
                self,
                "VpcLookup",
                vpc_name=VPC_NAME,
            )

        return sagemaker.CfnDomain(
            self,
            "MLSchoolDomain",
            auth_mode="IAM",
            default_user_settings=sagemaker.CfnDomain.UserSettingsProperty(
                execution_role=self.sagemaker_execution_role.role_arn,
            ),
            domain_name=f"mlschool-{self.stack_name.lower()}-domain",
            subnet_ids=[subnet.subnet_id for subnet in vpc.private_subnets],
            vpc_id=vpc.vpc_id,
        )

    def _create_default_user_profile(self) -> sagemaker.CfnUserProfile:
        """Create a default user profile for the SageMaker Domain."""
        # https://docs.aws.amazon.com/cdk/api/v2/docs/aws-cdk-lib.aws_sagemaker.CfnUserProfile.html

        return sagemaker.CfnUserProfile(
            self,
            "DefaultUserProfile",
            domain_id=self.sagemaker_domain.ref,
            user_profile_name="default",
            # user_settings=sagemaker.CfnUserProfile.UserSettingsProperty(
            #     execution_role=self.sagemaker_execution_role.role_arn,
            #     # Optional: Configure default instance types
            #     jupyter_server_app_settings=sagemaker.CfnUserProfile.JupyterServerAppSettingsProperty(
            #         default_resource_spec=sagemaker.CfnUserProfile.ResourceSpecProperty(
            #             instance_type="ml.t3.medium",
            #             sage_maker_image_arn=f"arn:aws:sagemaker:{self.region}:081325390199:image/datascience-1.0",
            #         )
            #     ),
            #     kernel_gateway_app_settings=sagemaker.CfnUserProfile.KernelGatewayAppSettingsProperty(
            #         default_resource_spec=sagemaker.CfnUserProfile.ResourceSpecProperty(
            #             instance_type="ml.t3.medium",
            #             sage_maker_image_arn=f"arn:aws:sagemaker:{self.region}:081325390199:image/datascience-1.0",
            #         )
            #     ),
            #     # Optional: Configure security groups (inherits from domain if not specified)
            #     # security_groups=["sg-12345678"]  # Add specific security groups if needed
            # ),
        )

    def _deploy_sample_data(self) -> None:
        """Deploy sample data to S3 if it exists."""

        try:
            # This will only work if the data file exists
            s3_deployment.BucketDeployment(
                self,
                "DeployPenguinsData",
                sources=[s3_deployment.Source.asset("./data")],
                destination_bucket=self.ml_bucket,
                destination_key_prefix="penguins/data/",
                prune=False,  # Don't delete existing objects
                retain_on_delete=False,
            )
        except Exception:
            # If data directory doesn't exist, create a placeholder
            print("Data directory not found, skipping data deployment")

    def _create_outputs(self) -> None:
        """Create CloudFormation outputs."""

        CfnOutput(
            self,
            "S3BucketName",
            value=self.ml_bucket.bucket_name,
            description="Name of the S3 bucket for ML data and artifacts",
            export_name=f"{self.stack_name}-S3BucketName",
        )

        # CfnOutput(
        #     self,
        #     "ECRRepositoryName",
        #     value=self.ecr_repository.repository_name,
        #     description="Name of the ECR repository for Docker images",
        #     export_name=f"{self.stack_name}-ECRRepositoryName",
        # )

        # CfnOutput(
        #     self,
        #     "ECRRepositoryUri",
        #     value=self.ecr_repository.repository_uri,
        #     description="URI of the ECR repository",
        #     export_name=f"{self.stack_name}-ECRRepositoryUri",
        # )

        CfnOutput(
            self,
            "SageMakerExecutionRoleArn",
            value=self.sagemaker_execution_role.role_arn,
            description="ARN of the SageMaker execution role",
            export_name=f"{self.stack_name}-SageMakerExecutionRoleArn",
        )

        CfnOutput(
            self,
            "SageMakerExecutionRoleName",
            value=self.sagemaker_execution_role.role_name,
            description="Name of the SageMaker execution role",
            export_name=f"{self.stack_name}-SageMakerExecutionRoleName",
        )

        CfnOutput(
            self,
            "SageMakerDomainId",
            value=self.sagemaker_domain.ref,
            description="SageMaker Domain ID",
            export_name=f"{self.stack_name}-SageMakerDomainId",
        )

        CfnOutput(
            self,
            "DefaultUserProfileName",
            value=self.default_user_profile.user_profile_name,
            description="Name of the default SageMaker user profile",
            export_name=f"{self.stack_name}-DefaultUserProfileName",
        )

        # Console links for easy access
        CfnOutput(
            self,
            "S3BucketConsoleLink",
            value=f"https://s3.console.aws.amazon.com/s3/buckets/{self.ml_bucket.bucket_name}",
            description="AWS Console link to the S3 bucket",
        )

        # SageMaker Domain Console Link
        CfnOutput(
            self,
            "SageMakerDomainConsoleLink",
            value=f"https://{self.region}.console.aws.amazon.com/sagemaker/home?region={self.region}#/studio/{self.sagemaker_domain.ref}",
            description="AWS Console link to the SageMaker domain",
        )

        # CfnOutput(
        #     self,
        #     "ECRConsoleLink",
        #     value=f"https://{self.region}.console.aws.amazon.com/ecr/repositories/private/{self.account}/{self.ecr_repository.repository_name}",
        #     description="AWS Console link to the ECR repository",
        # )

        CfnOutput(
            self,
            "IAMRoleConsoleLink",
            value=f"https://console.aws.amazon.com/iam/home?region={self.region}#/roles/{self.sagemaker_execution_role.role_name}",
            description="AWS Console link to the SageMaker execution role",
        )

        # Add notification infrastructure outputs
        CfnOutput(
            self,
            "NotificationQueueUrl",
            value=self.notification_queue.queue_url,
            description="URL of the notification SQS queue",
            export_name=f"{self.stack_name}-NotificationQueueUrl",
        )

        CfnOutput(
            self,
            "NotificationTopicArn",
            value=self.notification_topic.topic_arn,
            description="ARN of the notification SNS topic",
            export_name=f"{self.stack_name}-NotificationTopicArn",
        )

        CfnOutput(
            self,
            "NotificationLambdaArn",
            value=self.notification_lambda.function_arn,
            description="ARN of the notification Lambda function",
            export_name=f"{self.stack_name}-NotificationLambdaArn",
        )


###############
# --- App --- #
###############

# CDK App
app = cdk.App()

# Create the ML infrastructure stack
SagemakerMLStack(
    app,
    "sagemaker-ml",
    env=cdk.Environment(
        account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
        region=os.environ.get("CDK_DEFAULT_REGION"),
    ),
    description="SageMaker ML infrastructure for Palmer Penguins ML School project",
)

# Synthesize the app
app.synth()

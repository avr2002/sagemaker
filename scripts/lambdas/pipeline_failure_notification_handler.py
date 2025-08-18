import json
import os
from datetime import datetime, timezone
from typing import Any, Dict
from urllib import response

import boto3

try:
    from mypy_boto3_sagemaker.client import SageMakerClient
    from mypy_boto3_sns.client import SNSClient
except ImportError:
    ...

# Initialize AWS clients
sns_client: "SNSClient" = boto3.client("sns")
sm_client: "SageMakerClient" = boto3.client("sagemaker")

SNS_TOPIC_ARN = os.environ["SNS_TOPIC_ARN"]


# Tutorial: Using Lambda with Amazon SQS: https://docs.aws.amazon.com/lambda/latest/dg/with-sqs-example.html

# """
# Sample Payload that SQS receives from SageMaker CallbackStep

# {
#   "body": {
#     "token": "SYZ8ExZ3Sd",
#     "pipelineExecutionArn": "arn:aws:sagemaker:us-east-1:<acc-id>:pipeline/e2e-ml-pipeline/execution/wlaimaypb6hb",
#     "arguments": {
#       "model_package_group": "basic-penguins-model-group",
#       "baseline_accuracy": 0.8235294117647058,
#       "step_name": "evaluate-model",
#       "precision": 0.929561157796452,
#       "recall": 0.9215686274509803,
#       "accuracy": 0.9215686274509803
#     },
#     "status": "Executing"
#   }
# }
# """


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda function to handle SageMaker pipeline failure notifications.

    Processes SQS messages from SageMaker CallbackStep, sends SNS notifications,
    and responds back to SageMaker with success/failure status.
    """

    failed_records = []

    for record in event["Records"]:
        try:
            # Parse the SQS message
            message_body = json.loads(record["body"])

            # Extract SageMaker callback information
            callback_token = message_body.get("token")
            pipeline_execution_arn = message_body.get("pipelineExecutionArn")

            # Extract custom pipeline data (sent by CallbackStep)
            pipeline_data = message_body.get("arguments", {})
            step_name = pipeline_data["step_name"]

            # Format notification message
            notification_message = format_notification_message(pipeline_data, pipeline_execution_arn, step_name)

            # Send SNS notification
            send_sns_notification(notification_message, pipeline_data)

            # Send success response back to SageMaker
            send_pipeline_success(callback_token)

            print(f"Successfully processed notification for step: {step_name}")

        except Exception as e:
            print(f"Error processing record: {str(e)}")

            # Try to send failure response to SageMaker if we have the token
            try:
                message_body = json.loads(record["body"])
                callback_token = message_body.get("token")
                if callback_token:
                    send_pipeline_failure(callback_token, str(e))
            except Exception as callback_error:
                print(f"Failed to send callback failure: {str(callback_error)}")

            # Mark record as failed for SQS retry
            failed_records.append({"itemIdentifier": record["messageId"]})

    return {"batchItemFailures": failed_records}


def format_notification_message(pipeline_data: Dict[str, Any], execution_arn: str, step_name: str) -> str:
    """Format the notification message with pipeline details."""

    current_accuracy = pipeline_data.get("accuracy", "Unknown")
    baseline_accuracy = pipeline_data.get("baseline_accuracy", "Unknown")
    precision = pipeline_data.get("precision", "Unknown")
    recall = pipeline_data.get("recall", "Unknown")
    model_package_group = pipeline_data.get("model_package_group", "Unknown")

    # Extract execution ID from ARN
    execution_id = execution_arn.split("/")[-1] if execution_arn else "Unknown"

    message = f"""
🚨 ML Pipeline Failure Alert

Pipeline Execution: {execution_id}
Step: {step_name}
Time: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")} UTC

Model Performance:
• Current Accuracy: {current_accuracy}
• Required Baseline: {baseline_accuracy}
• Precision: {precision}
• Recall: {recall}
• Model Package Group: {model_package_group}

The model did not meet the required performance threshold and was not registered.

View execution details:
https://console.aws.amazon.com/sagemaker/home#/pipelines/executions/{execution_id}
    """.strip()

    return message


def send_sns_notification(message: str, pipeline_data: Dict[str, Any]) -> None:
    """Send notification to SNS topic."""

    subject = f"ML Pipeline Failure - Accuracy {pipeline_data.get('accuracy', 'Unknown')}"

    response = sns_client.publish(
        TopicArn=SNS_TOPIC_ARN,
        Message=message,
        Subject=subject,
    )
    print(f"SNS notification sent with MessageId: {response['MessageId']}")


def send_pipeline_success(callback_token: str) -> None:
    """Send success response back to SageMaker pipeline."""

    sm_client.send_pipeline_execution_step_success(
        CallbackToken=callback_token,
        # OutputParameters=[
        #     {"Name": "notification_status", "Value": "success"},
        #     {"Name": "notification_time", "Value": datetime.now(timezone.utc).isoformat()},
        # ],
    )


def send_pipeline_failure(callback_token: str, failure_reason: str) -> None:
    """Send failure response back to SageMaker pipeline."""

    sm_client.send_pipeline_execution_step_failure(
        CallbackToken=callback_token, FailureReason=f"Notification failed: {failure_reason}"
    )

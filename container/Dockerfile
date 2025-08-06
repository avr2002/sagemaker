# Use an ARM-compatible base image with Python
FROM --platform=linux/arm64 python:3.10-slim-bookworm

# Install system dependencies including build tools for compiling packages
RUN apt-get update && \
    apt-get install -y build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

# Install deps
# Install sagemaker-training toolkit that contains the common functionality necessary to create a container compatible with SageMaker AI and the Python SDK.
RUN pip install --no-cache-dir \
    tensorflow==2.12.0 \
    numpy \
    pandas \
    scikit-learn \
    comet-ml \
    sagemaker-training

ENV PYTHONUNBUFFERED=TRUE

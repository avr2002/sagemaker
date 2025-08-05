# https://docs.aws.amazon.com/sagemaker/latest/dg/sklearn.html

# Base image from SageMaker pre-built images
# FROM 683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3

FROM --platform=linux/amd64 python:3.10-slim-bookworm

# https://docs.aws.amazon.com/sagemaker/latest/dg/pre-built-docker-containers-scikit-learn-spark.html
# https://docs.aws.amazon.com/sagemaker/latest/dg-ecr-paths/ecr-us-east-1.html#sklearn-us-east-1

# Set the working directory
WORKDIR /opt/ml/processing/input/code

# Install Docker CLI and Docker Compose
RUN apt-get update && apt-get install -y \
    docker.io \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose \
    && chmod +x /usr/local/bin/docker-compose

# Copy/create bare minimum files needed to install dependencies
COPY pyproject.toml README.md ./
RUN mkdir -p src/penguins && touch src/penguins/__init__.py

# Install dependencies
RUN pip install --upgrade pip
RUN pip install hatchling
RUN pip install --editable .

# Copy the rest of the source code
COPY . .

# # Set Python path to include the source directory
# ENV PYTHONPATH="/opt/ml/processing/code:/opt/ml/processing/input/code:${PYTHONPATH}"

# # Set working directory for SageMaker processing
# WORKDIR /opt/ml/processing/code

ENTRYPOINT ["python", "pipeline.py"]

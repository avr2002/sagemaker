# Sagemaker Custom Inference Container
FROM python:3.10-slim-bookworm

# ref: https://docs.aws.amazon.com/sagemaker/latest/dg/build-multi-model-build-container.html
# ref: https://github.com/aws/amazon-sagemaker-examples/blob/4534bff4b5b5062af5789d98c4ddca01b0cb5d1f/advanced_functionality/multi_model_bring_your_own/container/Dockerfile#L3-L6

# Set a docker label to advertise multi-model support on the container
LABEL com.amazonaws.sagemaker.capabilities.multi-models=true
# Set a docker label to enable container to use SAGEMAKER_BIND_TO_PORT environment variable if present
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true

# Install system dependencies including build tools for compiling packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc curl ca-certificates \
    openjdk-17-jre-headless && \
    rm -rf /var/lib/apt/lists/*

# Configure Java (required by multi-model-server / MMS)
# https://github.com/awslabs/multi-model-server/tree/master
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:${PATH}"

# Set the working directory
WORKDIR /home/model-server/

COPY pyproject.toml uv.lock README.md /home/model-server/
RUN mkdir -p /home/model-server/src/penguins && \
    touch /home/model-server/src/penguins/__init__.py

# Install dependencies
RUN pip install --upgrade pip uv
RUN uv sync \
    --group inference \
    --no-install-project \
    --no-editable \
    --frozen --no-cache

# Recommended for running Multi Model Server on CPU hosts
RUN uv pip install --no-cache-dir mxnet

# Copy entrypoint script to the image
COPY src/penguins/inference/entrypoint.py /usr/local/bin/entrypoint.py
RUN chmod +x /usr/local/bin/entrypoint.py

RUN mkdir -p /home/model-server/

# Copy the default custom service file to handle incoming data and inference requests
COPY src/penguins/inference/model_handler.py /home/model-server/model_handler.py

# Copy the SageMaker Hosting 'serve' entrypoint script and make it executable
# Model entrypoint executable "serve" was not found in container PATH

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly.
# PYTHONDONTWRITEBYTECODE keeps Python from writing the .pyc files which are unnecessary in this case.
# Force Python to run in unbuffered mode - ensures stdout/stderr are not buffered
# This makes logs appear immediately in Docker, which is useful for debugging and monitoring
ENV PYTHONUNBUFFERED=TRUE \
PYTHONDONTWRITEBYTECODE=TRUE

ENV PATH="/home/model-server/.venv/bin:${PATH}"

# Define an entrypoint script for the docker image
ENTRYPOINT ["uv", "run", "--script", "/usr/local/bin/entrypoint.py"]

# Define command to be passed to the entrypoint
CMD ["serve"]
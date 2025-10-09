# Sagemaker Custom Training Container

FROM python:3.10-slim-bookworm

# Install system dependencies including build tools for compiling packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /opt/ml/code/

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy the SageMaker Training 'train' entrypoint script and make it executable
COPY containers/training/train /opt/ml/code/train
RUN chmod +x /opt/ml/code/train

COPY pyproject.toml uv.lock README.md /opt/ml/code/
RUN mkdir -p /opt/ml/code/src/penguins && \
    touch /opt/ml/code/src/penguins/__init__.py

# Install dependencies
RUN pip install --upgrade pip uv
RUN uv sync \
    --group training \
    --frozen --no-cache

# copy the rest of the code
COPY src/ /opt/ml/code/src/

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly.
# PYTHONDONTWRITEBYTECODE keeps Python from writing the .pyc files which are unnecessary in this case.

# Force Python to run in unbuffered mode - ensures stdout/stderr are not buffered
# This makes logs appear immediately in Docker, which is useful for debugging and monitoring
ENV PYTHONUNBUFFERED=TRUE \
    PYTHONDONTWRITEBYTECODE=TRUE

# Use the virtual environment automatically
ENV VIRTUAL_ENV=/opt/ml/code/.venv
# Update PATH
ENV PATH="/opt/ml/code/.venv/bin:$PATH"

# # The training script must be located in the /opt/ml/code directory.
# # The environment variable SAGEMAKER_PROGRAM defines which file inside the /opt/ml/code
# # directory to use as the training entry point. When training starts, the interpreter executes the entry point defined by SAGEMAKER_PROGRAM.
# # Python and shell scripts are both supported.
# ENV SAGEMAKER_PROGRAM train.py
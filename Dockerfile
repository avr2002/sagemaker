FROM --platform=linux/amd64 python:3.10-slim-bookworm
#  --platform=arm64

# Set the working directory
WORKDIR /opt/ml/processing/code

# Copy/create bare minimum files needed to install dependencies
COPY pyproject.toml .
RUN mkdir -p src/penguins && touch src/penguins/__init__.py

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --editable .

# Copy the rest of the source code
COPY . .

# RUN AWS_PROFILE="$AWS_PROFILE" python pipeline.py
ENTRYPOINT [ "python", "pipeline.py" ]

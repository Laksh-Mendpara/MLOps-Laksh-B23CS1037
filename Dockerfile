# -----------------------------------------------------------------------
# Task 2 Dockerfile - Development / Training environment
# -----------------------------------------------------------------------
# Base image: CUDA-enabled PyTorch for GPU training; falls back gracefully
# to CPU inside the container if no GPU is mounted.
# -----------------------------------------------------------------------

FROM python:3.11-slim

# System packages needed to build some pip wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Default: start an interactive shell so the user can run train / evaluate
CMD ["/bin/bash"]

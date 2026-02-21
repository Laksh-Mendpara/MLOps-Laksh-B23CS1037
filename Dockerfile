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

# Copy source code and notebook
COPY src/ ./src/
COPY ML_DL_Ops_Ass_3_Fine_Tuning_Classification.ipynb .

# Expose results directory as a volume so outputs persist on the host
VOLUME ["/app/results", "/app/logs", "/app/distilbert-reviews-genres"]

# Default: start an interactive shell so the user can run train / evaluate
CMD ["/bin/bash"]

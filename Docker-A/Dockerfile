# 1. Base Image: Use an official NVIDIA image with CUDA support
# Always match the CUDA version to the PyTorch/TF version you intend to use.
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# 2. Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# 3. Setup workspace
WORKDIR /workspace

# 4. Install essential system tools
# 'git' for cloning, 'htop' for monitoring, 'libgl1' for OpenCV
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    vim \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy requirements and install Python dependencies
# We copy requirements.txt FIRST to leverage Docker caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Run bash
CMD ["/bin/bash"]

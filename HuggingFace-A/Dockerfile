# 1. Base Image: NVIDIA PyTorch with CUDA 12.1
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# 2. Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# 3. Setup workspace
WORKDIR /workspace

# 4. Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    vim \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy all source files
COPY . .

# 7. Default command
CMD ["/bin/bash"]

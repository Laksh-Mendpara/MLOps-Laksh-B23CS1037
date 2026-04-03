FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies and uv
RUN apt-get update && apt-get install -y curl git && \
    curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Copy requirements
COPY pyproject.toml uv.lock ./
# Install to distinct virtual environment to avoid volume mount overriding .venv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
RUN uv sync

# Set environment
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH=/app/q1_vit_lora

# Default command
CMD ["python", "q1_vit_lora/run_all.py"]

FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/opt/conda/bin:$PATH

# Install system packages and Miniconda
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    git-lfs \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
RUN uv pip install runpod fast-flashtalk

COPY . .

# Run the serverless handler (RunPod looks for `handler.py`)
CMD ["python", "handler.py"]

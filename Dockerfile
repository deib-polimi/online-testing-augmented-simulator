# Use NVIDIA’s CUDA image to enable GPU support (cudnn8-devel variant for development)
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Suppress interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3-pip \
        git \
        build-essential \
        cmake \
        ffmpeg \
        libgl1-mesa-glx \
        wget \
        vulkan-tools \
        libvulkan1 \
        mesa-vulkan-drivers \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Uncomment these lines if you want to set python3.10 as the default python and pip versions
# RUN ln -s /usr/bin/python3.10 /usr/bin/python && \
#     ln -s /usr/bin/pip3 /usr/bin/pip

# Create directories for requirements and models
RUN mkdir /requirements
RUN mkdir /models

# Copy requirements file into container
COPY requirements.txt /requirements

# Install Python requirements with caching enabled
RUN --mount=type=cache,target=/root/.cache/pip pip install --upgrade pip
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /requirements/requirements.txt

# Copy the project into the container (assumes the Dockerfile is in the repository root)
COPY . online-testing-augmented-simulator

# Set working directory and environment variables
WORKDIR online-testing-augmented-simulator
ENV PYTHONPATH=/online-testing-augmented-simulator
ENV BASE_DIR=/online-testing-augmented-simulator
ENV MODEL_DIR=/online-testing-augmented-simulator/models
ENV RESULTS_DIR=/online-testing-augmented-simulator/results

# Make sure our experiment script is executable
RUN chmod +x run_experiments.sh

# By default, run the experiment script
CMD ["./run_experiments.sh"]
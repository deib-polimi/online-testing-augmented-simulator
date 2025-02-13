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
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
# RUN ln -s /usr/bin/python3.10 /usr/bin/python && \
#     ln -s /usr/bin/pip3 /usr/bin/pip

# Create a working directory and copy the project
RUN git clone https://github.com/deib-polimi/online-testing-augmented-simulator

WORKDIR online-testing-augmented-simulator

# Install Python requirements
# If 'requirements.txt' is in the root of the repo, run:
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Install Python requirements
# If 'requirements.txt' is in the root of the repo, run:
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Make sure our experiment script is executable
RUN chmod +x /app/run_experiments.sh

# By default, run the experiment script
CMD ["/app/run_experiments.sh"]
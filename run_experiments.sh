#!/usr/bin/env bash
#
# run_experiments.sh
#
# This script runs all experiments as described in the repository. It assumes that you have
# cloned the repository and that all subdirectories (e.g., 'scripts/offline', 'scripts/online', etc.)
# are in place.
#
# When executed inside the Docker container, it will:
#   1. Optionally extract any needed archives (e.g., pretrained models, simulator binaries)
#   2. Prepare necessary directories and move model files if they have not been set up yet.
#   3. Run the online experiments sequentially.
#
# Note:
#   - The simulator extraction is performed only if its target directory is not present.
#   - Directories for models are created only if they do not exist, and model checkpoint files
#     are moved only if they are not already in place.
#   - Ensure that the environment variable BASE_DIR is set; if not, it defaults to "/".
#

# Exit immediately if a command exits with a non-zero status and treat unset variables as errors
set -e
set -u

# Set BASE_DIR: use environment variable BASE_DIR if exists; otherwise default to "/"
BASE_DIR="${BASE_DIR:-/}"

echo "===== Starting experiments for online-testing-augmented-simulator ====="

# -------------------------------
# Step 1: Extract Simulator Archive
# -------------------------------
# Check if the simulator has already been extracted (assumes extraction creates a directory named "udacity-linux")
if [ ! -d "udacity-linux" ]; then
  echo "Extracting udacity-linux simulator archive..."
  tar -xJf udacity-linux.tar.xz
else
  echo "Simulator archive already extracted; skipping extraction."
fi

# -------------------------------
# Step 2: Extract Pretrained Models Archive
# -------------------------------
# Check if the archive has been extracted by verifying one of the expected model files exists in the target location.
MODEL_TARGET_DIR="$BASE_DIR/models/online-testing"
DAVE2_MODEL="${MODEL_TARGET_DIR}/dave2/dave2.ckpt"

if [ ! -f "${DAVE2_MODEL}" ]; then
  echo "Extracting pretrained models archive..."
  tar -xJf pretrained-udacity-models.tar.xz

  # Create target directories for each model if they do not exist.
  mkdir -p "${MODEL_TARGET_DIR}/dave2/"
  mkdir -p "${MODEL_TARGET_DIR}/chauffeur/"
  mkdir -p "${MODEL_TARGET_DIR}/epoch/"
  mkdir -p "${MODEL_TARGET_DIR}/vit/"

  # Move model files into their respective directories if they exist in the current directory.
  [ -f "dave2.ckpt" ] && mv dave2.ckpt "${MODEL_TARGET_DIR}/dave2/"
  [ -f "chauffeur.ckpt" ] && mv chauffeur.ckpt "${MODEL_TARGET_DIR}/chauffeur/"
  [ -f "epoch.ckpt" ] && mv epoch.ckpt "${MODEL_TARGET_DIR}/epoch/"
  [ -f "vit.ckpt" ] && mv vit.ckpt "${MODEL_TARGET_DIR}/vit/"
else
  echo "Pretrained models already set up in ${MODEL_TARGET_DIR}; skipping extraction and file moves."
fi

# -------------------------------
# Step 3: Run Online Experiments
# -------------------------------
# Execute the experiment scripts in sequence. Adjust or add scripts as needed.
echo "Running nominal experiment..."
python3 scripts/online/nominal.py

echo "Running instructpix2pix experiment..."
python3 scripts/online/instructpix2pix.py

echo "Running stable diffusion inpainting experiment..."
python3 scripts/online/stable_diffusion_inpainting.py

echo "Running stable diffusion inpainting with controlnet refining experiment..."
python3 scripts/online/stable_diffusion_inpainting_controlnet_refining.py

echo "===== All experiments completed successfully! ====="

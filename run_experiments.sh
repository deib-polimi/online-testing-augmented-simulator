#!/usr/bin/env bash
#
# run_experiments.sh
#
# This script runs all experiments as described in the repository.
# It assumes that you have cloned the repository and have all subfolders
# and scripts in place (e.g., 'scripts/offline', 'scripts/online', etc.).
#
# When this script is executed inside the Docker container, it will:
#  1. (Optionally) extract any needed archives (pretrained models, etc.)
#  2. Run the offline experiments
#  3. Run the online experiments (requires the Udacity simulator to be running)
#  4. Run OOD analysis experiments
#  5. Run timing analysis

set -e  # Exit immediately if a command exits with a non-zero status
set -u  # Treat unset variables as an error

echo "===== Starting experiments for online-testing-augmented-simulator ====="

# Optional Step: Extract archives if present
if [[ -f pretrained-udacity-models.tar.xz ]]; then
  echo "Extracting pretrained models..."
  tar -xf pretrained-udacity-models.tar.xz
fi

if [[ -f udacity-gym-env.tar.xz ]]; then
  echo "Extracting Udacity gym environment..."
  tar -xf udacity-gym-env.tar.xz
fi

# 1. Offline augmentation experiments
echo "----- Running offline augmentation experiments -----"
python3 scripts/offline/instructpix2pix.py
python3 scripts/offline/stable_diffusion_inpainting.py
python3 scripts/offline/stable_diffusion_inpainting_controlnet_refining.py

# 2. Online testing experiments (requires the Udacity simulator)
echo "----- Running online testing experiments -----"
echo "Ensure the Udacity simulator is already running!"
python3 scripts/online/001_error_analysis.py
python3 scripts/online/001_error_analysis_cyclegan_inpaint.py
python3 scripts/online/001_error_analysis_cyclegan_instruct.py

# 3. Out-of-distribution (OOD) analysis
echo "----- Running OOD analysis experiments -----"
python3 scripts/ood/clustering.py
python3 scripts/ood/compute_reconstruction_error.py
python3 scripts/ood/compute_reconstruction_error_simulator.py

# 4. Transformation timing analysis
echo "----- Running timing analysis experiments -----"
python3 scripts/timing/cyclegan.py
python3 scripts/timing/dave.py
python3 scripts/timing/instructpix2pix.py

echo "===== All experiments completed successfully! ====="

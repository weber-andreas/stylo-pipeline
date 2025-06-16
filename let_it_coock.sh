#!/bin/bash
# This script sets up the environment for the project by installing necessary packages and setting environment variables.

echo "Setting up the environment..."
# Load conda functions into the shell
source ~/.bashrc
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n stylo2 python=3.10 -y
conda activate stylo2


# First install pytorch
echo "Installing PyTorch..."
pip install torch==2.0.0

# # Install other environment dependencies
echo "Installing requirements of building blocks..."
conda env update -f environment.yaml

# Install local wheels like detectro2 and lang-sam (workaround)
# echo "Installing local packages..."
pip install -e ./building_blocks/detectron2/
pip install -e ./building_blocks/detectron2/projects/DensePose
pip install -e ./building_blocks/lang-segment-anything/
pip install huggingface_hub==0.25.2
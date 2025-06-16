#!/bin/bash
# This script sets up the environment for the project by installing necessary packages and setting environment variables.

# echo "Setting up the environment..."
conda init
# conda create -n stylo python=3.10 -y
conda activate stylo

# First install pytorch
echo "Installing PyTorch..."
pip install torch==2.5

# Install other environment dependencies
echo "Installing conda dependencies..."
conda env update -f ./environment.yml

# Install local wheels like detectro2 and lang-sam (workaround)
# echo "Installing local packages..."
# pip install -e ./building_blocks/detectron2/
# pip install -e ./building_blocks/detectron2/projects/DensePose
# pip install -e ./building_blocks/lang-segment-anything/

#!/bin/bash

# Setup script for Ship Detection Project
# This script automates the installation process

set -e  # Exit on error

echo "=========================================="
echo "Ship Detection Project Setup"
echo "=========================================="

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "Conda found. Creating environment..."
    conda create -n ships_detection python=3.11.9 -y
    echo "Activating environment..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate ships_detection
else
    echo "Conda not found. Using system Python..."
    echo "Please ensure Python 3.11+ is installed."
fi

# Navigate to RT-DETRv4 directory
cd RT-DETRv4

# Install PyTorch (adjust CUDA version as needed)
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install project dependencies
echo "Installing project dependencies..."
pip install -r requirements.txt

# Install inference dependencies
echo "Installing inference dependencies..."
pip install -r tools/inference/requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p pretrain
mkdir -p outputs
mkdir -p examples/images
mkdir -p examples/results

# Download instructions for teacher model
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Download DINOv3 teacher model:"
echo "   Visit: https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/"
echo "   Download: dinov3_vitb16_pretrain_lvd1689m.pth"
echo "   Place in: RT-DETRv4/pretrain/"
echo ""
echo "2. Prepare your dataset in COCO format"
echo ""
echo "3. Update config files:"
echo "   - configs/dataset/ships_open_vocab.yml"
echo "   - configs/rtv4/rtv4_hgnetv2_m_ships_open_vocab.yml"
echo ""
echo "4. Start training:"
echo "   python train.py -c configs/rtv4/rtv4_hgnetv2_m_ships_open_vocab.yml --use-amp"
echo ""
echo "For more information, see README.md"
echo "=========================================="


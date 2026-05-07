#!/bin/bash
# One-click setup script for Oasis 500M + CogVideoX on Alibaba Cloud GPU server
# Tested on: Ubuntu 20.04/22.04 with NVIDIA driver pre-installed

set -e

echo "============================================"
echo "  Assignment 3: World Model Setup Script"
echo "============================================"

# Step 1: System check
echo ""
echo "[1/6] Checking GPU..."
nvidia-smi || { echo "ERROR: No NVIDIA GPU detected. Make sure you rented a GPU instance."; exit 1; }

# Step 2: Create working directory
echo ""
echo "[2/6] Creating workspace..."
mkdir -p ~/world_model && cd ~/world_model

# Step 3: Clone Oasis
echo ""
echo "[3/6] Cloning open-oasis..."
if [ ! -d "open-oasis" ]; then
    git clone https://github.com/etched-ai/open-oasis.git
fi
cd open-oasis

# Step 4: Install Python dependencies
echo ""
echo "[4/6] Installing Python packages..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install einops diffusers timm av pillow numpy

# Step 5: Download model weights
echo ""
echo "[5/6] Downloading Oasis 500M weights from HuggingFace..."
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
import os

save_dir = './weights'
os.makedirs(save_dir, exist_ok=True)

print('Downloading oasis500m.safetensors...')
hf_hub_download(repo_id='Etched/oasis-500m', filename='oasis500m.safetensors', local_dir=save_dir)

print('Downloading vit-l-20.safetensors (VAE)...')
hf_hub_download(repo_id='Etched/oasis-500m', filename='vit-l-20.safetensors', local_dir=save_dir)

print('Done! Weights saved to ./weights/')
"

# Step 6: Install CogVideoX dependencies (for bonus comparison)
echo ""
echo "[6/6] Installing CogVideoX dependencies..."
pip install transformers accelerate sentencepiece

echo ""
echo "============================================"
echo "  Setup complete!"
echo "  Oasis weights: ~/world_model/open-oasis/weights/"
echo "  Next: run 'python generate_demo.py' to generate videos"
echo "============================================"

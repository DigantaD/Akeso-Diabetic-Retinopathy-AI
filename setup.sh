#!/bin/bash

set -e

# -----------------------------
# ⚙️ System-Level Dependencies
# -----------------------------
echo "Installing APT packages..."
sudo apt update && sudo apt install -y \
    git \
    wget \
    unzip \
    ffmpeg \
    libgl1

# -----------------------------
# 🧪 Python Requirements
# -----------------------------
echo "Installing Python dependencies from requirements.txt..."
pip3 install --upgrade pip
pip3 install -r requirements.txt

# -----------------------------
# 🧬 Segment Anything Git Clone
# -----------------------------
echo "Cloning Segment Anything repo..."
git clone https://github.com/facebookresearch/segment-anything.git || true
cd segment-anything && pip3 install -e . && cd ..

# -----------------------------
# ⬇️ Pretrained SAM Model
# -----------------------------
echo "Downloading SAM ViT-B pretrained model..."
mkdir -p pretrained
wget -nc https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P pretrained/

echo "✅ Setup complete. You're ready to run Akeso!"
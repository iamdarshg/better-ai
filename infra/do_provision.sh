#!/bin/bash
set -e

# Better AI - DigitalOcean GPU Droplet Provisioning Script
# This script is intended to be used as User Data (cloud-init) when creating a Droplet.
# Targeted for Ubuntu 22.04 LTS on DigitalOcean GPU Droplets (H100).

echo "🚀 Starting Better AI Provisioning..."

# 1. Update and install basic dependencies
apt-get update
apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    git \
    wget

# 2. Install Docker
if ! command -v docker &> /dev/null; then
    echo "📦 Installing Docker..."
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg

    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
      tee /etc/apt/sources.list.d/docker.list > /dev/null

    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi

# 3. Install NVIDIA Container Toolkit
if ! command -v nvidia-ctk &> /dev/null; then
    echo "🛠️ Installing NVIDIA Container Toolkit..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    apt-get update
    apt-get install -y nvidia-container-toolkit

    # Configure Docker to use NVIDIA runtime
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
fi

# 4. Mount Block Storage Volume (if attached)
# DigitalOcean volumes are typically mapped to /dev/disk/by-id/scsi-0DO_Volume_VOLUME_NAME
# We'll wait a few seconds for the volume to appear if it was just attached.
sleep 5
VOLUME_DEV=$(ls /dev/disk/by-id/scsi-0DO_Volume_* 2>/dev/null | head -n 1)
if [ -n "$VOLUME_DEV" ]; then
    echo "💾 DigitalOcean Volume detected at $VOLUME_DEV"
    MOUNT_POINT="/app/checkpoints"
    mkdir -p "$MOUNT_POINT"

    # Format the volume if it doesn't have a filesystem
    if ! blkid "$VOLUME_DEV" > /dev/null; then
        echo "✨ Formatting volume $VOLUME_DEV..."
        mkfs.ext4 -F "$VOLUME_DEV"
    fi

    # Mount and add to fstab for persistence
    if ! grep -q "$MOUNT_POINT" /etc/fstab; then
        echo "🔗 Mounting volume to $MOUNT_POINT..."
        mount "$VOLUME_DEV" "$MOUNT_POINT"
        echo "$VOLUME_DEV $MOUNT_POINT ext4 defaults,nofail 0 2" >> /etc/fstab
    fi
else
    echo "ℹ️ No DigitalOcean Volume detected. Checkpoints will be stored on the local disk."
fi

# 5. Clone the repository
REPO_DIR="/app/better-ai"
if [ ! -d "$REPO_DIR" ]; then
    echo "📂 Cloning Better AI repository..."
    mkdir -p /app
    git clone https://github.com/iamdarshg/better-ai.git "$REPO_DIR"
fi

cd "$REPO_DIR"

# Ensure checkpoints directory exists and is linked if volume mounted
if [ -d "/app/checkpoints" ] && [ ! -L "checkpoints" ]; then
    rm -rf checkpoints
    ln -s /app/checkpoints checkpoints
fi

# 6. Build and Verify Environment
echo "🏗️ Building Better AI Docker image..."
docker compose build

echo "🧪 Running Smoke Test..."
# Run a quick check using the built image to ensure it can see the GPU and load torch
docker compose run --rm better-ai python -c "import torch; print(f'Better AI Smoke Test Passed! CUDA available: {torch.cuda.is_available()}')"

# 7. Final Verification
echo "✅ Provisioning Complete!"
echo "--------------------------------------------------"
nvidia-smi

echo "=================================================="
echo "Better AI environment is ready."
echo "To start training, run:"
echo "cd $REPO_DIR && docker compose up -d"

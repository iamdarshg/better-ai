#!/bin/bash

# Quick setup script for DeepSeek MoE training

echo "🚀 Setting up DeepSeek MoE Training Environment"
echo "=================================================="

# Check Python version
python_version=$(python3 --version 2>&1)
echo "🐍 Python: $python_version"

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "🖥️  NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1
else
    echo "⚠️  No NVIDIA GPU detected - training will use CPU"
fi

# Install requirements
echo ""
echo "📦 Installing requirements..."
pip install -r requirements.txt

# Create directories
echo ""
echo "📁 Creating directories..."
mkdir -p checkpoints
mkdir -p logs
mkdir -p dataset_cache

# Download small test dataset (optional)
echo ""
echo "📥 Setting up test data..."
python3 -c "
import os
try:
    from datasets import load_dataset
    print('Downloading small test dataset...')
    dataset = load_dataset('HuggingFaceH4/CodeAlpaca_20K', split='train[:100]')
    print(f'Test dataset loaded: {len(dataset)} samples')
    print('✅ Dataset setup successful!')
except Exception as e:
    print(f'❌ Dataset setup failed: {e}')
    print('Will use synthetic data instead')
"

echo ""
echo "🎯 Setup complete! You can now run training with:"
echo ""
echo "# Basic training with real data:"
echo "python3 train_moe.py"
echo ""
echo "# Python-only training:"
echo "python3 train_moe.py --python-only"
echo ""
echo "# Synthetic data (faster):"
echo "python3 train_moe.py --use-synthetic"
echo ""
echo "# Custom configuration:"
echo "python3 train_moe.py --max-steps 5000 --batch-size 2"
echo ""
echo "📊 Monitor training with:"
echo "tensorboard --logdir logs"
echo ""
echo "🎉 Happy training!"
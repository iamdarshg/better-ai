#!/usr/bin/env python3
"""
Quick setup script for the DeepSeek V3.2-inspired MoE model training
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error {description}: {e}")
        print(f"Output: {e.output}")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("🚀 Setting up DeepSeek V3.2-inspired MoE Model Training")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    dependencies = [
        ("torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118", 
         "PyTorch with CUDA support"),
        ("transformers", "Hugging Face Transformers"),
        ("datasets", "Hugging Face Datasets"),
        ("tokenizers", "Hugging Face Tokenizers"),
        ("accelerate", "Hugging Face Accelerate"),
        ("wandb", "Weights & Biases for tracking"),
        ("numpy", "NumPy"),
        ("tqdm", "Progress bars"),
        ("matplotlib", "Plotting"),
        ("seaborn", "Statistical plotting"),
    ]
    
    success = True
    for cmd, desc in dependencies:
        if not run_command(f"./bin/pip install {cmd}", desc):
            success = False
    
    if not success:
        print("\n⚠️  Some dependencies failed to install.")
        print("You can try installing manually or continue without them.")
    
    # Create necessary directories
    directories = ["checkpoints", "logs", "data"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    print("\n" + "=" * 60)
    print("🎉 Setup completed!")
    print("\n📚 Quick Start Guide:")
    print("1. Train with synthetic data:")
    print("   python train_moe_working.py --epochs 3")
    print()
    print("2. Train with real datasets (requires Hugging Face auth):")
    print("   huggingface-cli login")
    print("   python train_moe_real_data.py --num-train-samples 1000 --epochs 5")
    print()
    print("3. Full configuration:")
    print("   python train_moe_real_data.py \\")
    print("     --num-experts 8 \\")
    print("     --hidden-size 512 \\")
    print("     --batch-size 8 \\")
    print("     --learning-rate 1e-4 \\")
    print("     --epochs 10")
    print()
    print("📖 For more details, see the TODO.md file")

if __name__ == "__main__":
    main()

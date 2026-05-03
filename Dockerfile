# Base image from NVIDIA PyTorch container as specified in issue #27
# nvcr.io/nvidia/pytorch:24.03-py3 contains:
# - PyTorch 2.3.0
# - CUDA 12.4
# - Python 3.10
# - Triton (pre-installed)
FROM nvcr.io/nvidia/pytorch:24.03-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Optimize image size and fix dependency conflicts in a single layer
# 1. We use a constraints file to ensure the pre-installed torch/torchvision/triton
#    versions are NOT overwritten by newer, potentially incompatible versions from PyPI.
# 2. We remove them from requirements.txt to be doubly sure.
# 3. We use --no-cache-dir to avoid bloating the image.
# 4. We perform aggressive cleanup of unused CUDA components and package artifacts.
RUN pip freeze > /tmp/constraints.txt && \
    sed -i '/torch/d' requirements.txt && \
    sed -i '/torchvision/d' requirements.txt && \
    sed -i '/triton/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt -c /tmp/constraints.txt && \
    # Optional GPU extras
    (pip install --no-cache-dir "flash-attn>=2.5.0" --no-build-isolation || echo "Optional flash-attn installation failed") && \
    # Aggressive cleanup to keep image size under 20GB
    # Remove large static libraries and unused CUDA components (saves GBs)
    find /usr/local/cuda -name "*.a" -delete || true && \
    # Remove documentation and examples from packages
    find /usr/local/lib/python3.10/dist-packages -name "docs" -type d -exec rm -rf {} + || true && \
    find /usr/local/lib/python3.10/dist-packages -name "examples" -type d -exec rm -rf {} + || true && \
    # Clean up apt and temporary files
    rm -rf /root/.cache/pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm /tmp/constraints.txt

# Copy the rest of the source code
COPY . .

# Build-time verification: ensures the environment is correctly set up
# and better_ai is importable.
RUN python -c "import torch; import better_ai; print('Package import successful')"
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Default command for the container
CMD ["python", "train_enhanced.py", "--stage", "pretrain", "--test"]

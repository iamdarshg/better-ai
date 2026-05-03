# Base image from NVIDIA PyTorch container as specified in issue #27
# nvcr.io/nvidia/pytorch:24.03-py3 contains:
# - PyTorch 2.3.0 (with CUDA 12.4 matching)
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
# 1. We remove torch, torchvision, and triton from requirements.txt to preserve
#    the optimized, CUDA-compatible versions pre-installed in the NGC image.
# 2. We use --no-cache-dir to avoid bloating the image with temporary files.
# 3. We clean up large unnecessary directories (like documentation and tests)
#    from the system and pip installation.
RUN sed -i '/torch/d' requirements.txt && \
    sed -i '/torchvision/d' requirements.txt && \
    sed -i '/triton/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt && \
    # Optional GPU extras - using --no-build-isolation to avoid dependency re-checks
    (pip install --no-cache-dir "flash-attn>=2.5.0" || echo "Optional flash-attn installation failed") && \
    # Aggressive cleanup to keep image size under 20GB
    # Remove tests, docs, and cached files from all packages
    find /usr/local/lib/python3.10/dist-packages -name "tests" -type d -exec rm -rf {} + || true && \
    find /usr/local/lib/python3.10/dist-packages -name "testing" -type d -exec rm -rf {} + || true && \
    find /usr/local/lib/python3.10/dist-packages -name "docs" -type d -exec rm -rf {} + || true && \
    find /usr/local/lib/python3.10/dist-packages -name "__pycache__" -type d -exec rm -rf {} + || true && \
    # Remove large static libraries if any
    find /usr/local/lib/python3.10/dist-packages -name "*.a" -delete || true && \
    rm -rf /root/.cache/pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the rest of the source code
COPY . .

# Build-time verification: ensures the environment is correctly set up
# and better_ai is importable.
RUN python -c "import torch; import better_ai; print('Package import successful')"
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Default command for the container
CMD ["python", "train_enhanced.py", "--stage", "pretrain", "--test"]

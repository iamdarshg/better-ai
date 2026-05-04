# Base image from NVIDIA PyTorch container as specified in issue #27
FROM nvcr.io/nvidia/pytorch:24.03-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set working directory
WORKDIR /app

# Copy optimized requirements
COPY requirements-docker.txt .

# Optimize image size in a single layer
# 1. Install missing app dependencies using constraints to protect NGC stack.
# 2. Perform extremely aggressive cleanup of non-runtime assets.
RUN pip freeze | grep -v "@ file://" | grep -v "/rapids/" > /tmp/constraints.txt && \
    pip install --no-cache-dir -r requirements-docker.txt -c /tmp/constraints.txt && \
    # EXTREMELY Aggressive cleanup to meet the <20GB requirement
    # Remove all CUDA static libraries (saves multiple GBs)
    find /usr/local/cuda -name "*.a" -delete || true && \
    find /usr/lib/x86_64-linux-gnu -name "*.a" -delete || true && \
    # Remove Nsight, samples, documentation and redundant binaries
    rm -rf /usr/local/cuda/nsight* /usr/local/cuda/samples /usr/local/cuda/doc /usr/local/cuda/bin/nvvp && \
    # Remove documentation and examples from Python packages
    find /usr/local/lib/python3.10/dist-packages -name "docs" -type d -exec rm -rf {} + || true && \
    find /usr/local/lib/python3.10/dist-packages -name "examples" -type d -exec rm -rf {} + || true && \
    # Remove caches
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

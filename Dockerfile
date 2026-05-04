# Base image from NVIDIA PyTorch container as specified in issue #27
# nvcr.io/nvidia/pytorch:24.03-py3 contains:
# - PyTorch 2.3.0
# - CUDA 12.4
# - Python 3.10
# - Triton 2.3.0 (pre-installed)
# - Flash-Attn 2.4.2 (pre-installed)
FROM nvcr.io/nvidia/pytorch:24.03-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set working directory
WORKDIR /app

# Copy optimized requirements for Docker
COPY requirements-docker.txt .

# Install app dependencies with constraints to protect NGC stack
# 1. Capture NGC's pre-installed package versions as constraints.
# 2. Filter constraints to remove local file references for portability.
# 3. Install requirements using these constraints to prevent accidental replacements
#    of optimized Torch/CUDA/Flash-Attn/Triton components.
RUN pip freeze | grep -v "@ file://" > /tmp/constraints.txt && \
    pip install --no-cache-dir -r requirements-docker.txt -c /tmp/constraints.txt && \
    rm /tmp/constraints.txt

# Copy the rest of the source code
COPY . .

# Build-time verification: ensures the environment is correctly set up
# and better_ai is importable using the pre-installed NGC stack.
RUN python -c "import torch; import better_ai; print(f'PyTorch {torch.__version__} with better_ai imported successfully')"

# Default command for the container
CMD ["python", "train_enhanced.py", "--stage", "pretrain", "--test"]

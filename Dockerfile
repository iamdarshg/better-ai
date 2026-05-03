# Base image from NVIDIA PyTorch container as specified in issue #27
# nvcr.io/nvidia/pytorch:24.03-py3 contains:
# - PyTorch 2.2.1
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

# Install Python dependencies from requirements.txt
# We remove torch and torchvision from requirements.txt to preserve
# the optimized, CUDA-compatible versions pre-installed in the NGC image.
RUN sed -i '/torch/d' requirements.txt && \
    sed -i '/torchvision/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# Install GPU-optional extras requested in the issue
# flash-attn is version-sensitive; we install it separately to avoid conflicts.
# Triton is typically already in the NGC image, but we ensure it's available.
RUN pip install --no-cache-dir "flash-attn>=2.5.0" || echo "Optional flash-attn installation failed"

# Copy the rest of the source code
COPY . .

# Build-time verification: ensures the environment is correctly set up
# and better_ai is importable.
RUN python -c "import torch; import better_ai; print('Package import successful')"
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Default command for the container
CMD ["python", "train_enhanced.py", "--stage", "pretrain", "--test"]

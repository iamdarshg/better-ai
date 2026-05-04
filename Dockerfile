# Stage 1: Build and Cleanup
FROM nvcr.io/nvidia/pytorch:24.03-py3 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy optimized requirements
COPY requirements-docker.txt .

# 1. Protect the NGC stack.
#    We create a constraints file that pins EXACTLY what is currently installed.
#    This prevents pip from replacing the optimized NGC torch/cuda/triton stack.
# 2. Install missing app dependencies.
# 3. Perform extremely aggressive cleanup of non-runtime assets.
RUN pip freeze > /tmp/full_freeze.txt && \
    # Filter out local file references and absolute paths that break constraints
    grep -v "@ file://" /tmp/full_freeze.txt | grep -v "/rapids/" > /tmp/constraints.txt && \
    # Install app-level dependencies using the constraints to pin the NGC stack.
    pip install --no-cache-dir -r requirements-docker.txt -c /tmp/constraints.txt && \
    # VERIFICATION: Ensure torch was not uninstalled or replaced
    python -c "import torch; assert 'nv' in torch.__version__, f'NGC torch replaced! Current version: {torch.__version__}'" && \
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
    rm /tmp/constraints.txt /tmp/full_freeze.txt

# Stage 2: Final Flattened Image
# Using 'scratch' and copying from 'builder' is the only way to truly flatten and
# EXCLUDE files from the final image that were in the original base image.
FROM scratch

# Set essential environment variables for NGC stack
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PATH=/usr/local/npt/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/lib:/usr/lib/x86_64-linux-gnu
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Copy the entire cleaned filesystem from builder
COPY --from=builder / /

WORKDIR /app

# Copy the rest of the source code
COPY . .

# Final Build-time verification
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); assert 'nv' in torch.__version__"
RUN python -c "import better_ai; print('Package import successful')"

# Default command for the container
CMD ["python", "train_enhanced.py", "--stage", "pretrain", "--test"]

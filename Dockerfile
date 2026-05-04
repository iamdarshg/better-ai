# Stage 1: Build and Cleanup
FROM nvcr.io/nvidia/pytorch:24.03-py3 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy optimized requirements
COPY requirements-docker.txt .

# SURGICAL DEPENDENCY INSTALLATION
# This ensures we NEVER touch the NGC-provided torch/cuda stack.
# 1. Generate a report of all required dependencies (including transitive ones).
RUN pip install --upgrade pip && \
    pip install --dry-run --report /tmp/report.json -r requirements-docker.txt

# 2. Filter out any package that is already provided by NGC or belongs to the GPU stack.
RUN python3 <<EOF
import json
import os

with open("/tmp/report.json") as f:
    report = json.load(f)

exclude = {"torch", "torchvision", "torchtext", "triton", "flash-attn", "tensorrt", "nvidia", "cuda"}
to_install = []

for req in report["install"]:
    name = req["metadata"]["name"].lower()
    version = req["metadata"]["version"]

    # Check if the package or any part of its name is in the exclusion list
    should_exclude = any(ex in name for ex in exclude)

    if not should_exclude:
        to_install.append(f"{name}=={version}")

with open("/tmp/filtered_reqs.txt", "w") as f:
    f.write("\n".join(to_install))

print(f"Filtered {len(to_install)} dependencies for installation.")
EOF

# 3. Install only the missing app dependencies without checking sub-dependencies.
# 4. Perform extremely aggressive cleanup to meet <20GB limit.
RUN pip install --no-cache-dir --no-deps -r /tmp/filtered_reqs.txt && \
    # VERIFICATION: Ensure torch was not uninstalled or replaced
    python -c "import torch; assert 'nv' in torch.__version__, f'NGC torch replaced! Current version: {torch.__version__}'" && \
    # Remove all CUDA static libraries (saves multiple GBs)
    find /usr/local/cuda -name "*.a" -delete || true && \
    find /usr/lib/x86_64-linux-gnu -name "*.a" -delete || true && \
    rm -rf /usr/local/cuda/nsight* /usr/local/cuda/samples /usr/local/cuda/doc /usr/local/cuda/bin/nvvp && \
    find /usr/local/lib/python3.10/dist-packages -name "docs" -type d -exec rm -rf {} + || true && \
    find /usr/local/lib/python3.10/dist-packages -name "examples" -type d -exec rm -rf {} + || true && \
    rm -rf /root/.cache/pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm /tmp/report.json /tmp/filtered_reqs.txt

# Stage 2: Final Flattened Image
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
# Ensures PyTorch is optimized build AND Transformers can actually use it
RUN python -c "import torch, transformers; \
print(f'PyTorch version: {torch.__version__}'); \
print(f'Transformers version: {transformers.__version__}'); \
assert 'nv' in torch.__version__, 'NGC torch replaced!'; \
assert transformers.is_torch_available(), 'Transformers cannot see PyTorch! Check version compatibility.'; \
print('Container verification successful')"

# Default command for the container
CMD ["python", "train_enhanced.py", "--stage", "pretrain", "--test"]

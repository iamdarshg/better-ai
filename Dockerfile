# Base image from NVIDIA PyTorch container as specified in issue #27
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
RUN pip install --no-cache-dir -r requirements.txt

# Install GPU-optional extras requested in the issue
RUN pip install --no-cache-dir flash-attn triton

# Copy the rest of the source code
COPY . .

# Verify the installation (will fail the build if import fails)
RUN python -c "import torch; import better_ai; print('better_ai imported successfully')"

# Default command for the container
CMD ["python", "train_enhanced.py", "--stage", "pretrain", "--test"]

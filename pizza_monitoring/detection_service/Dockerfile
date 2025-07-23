# Use standard Python image instead of CUDA for better compatibility
FROM python:3.9-slim

# System dependencies - minimal installation
RUN apt-get clean && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
    git libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/

# Install Python dependencies in smaller batches to avoid memory issues
RUN pip install --upgrade pip
RUN pip install opencv-python==4.8.1.78 numpy==1.24.3 pika==1.3.2
RUN pip install torch==2.3.1 torchvision==0.18.1
RUN pip install ultralytics==8.3.166

# Copy project code
COPY . /app

# Create directory for weights
RUN mkdir -p /app/weights

# Default command to run your detection logic
CMD ["python", "detector.py"]

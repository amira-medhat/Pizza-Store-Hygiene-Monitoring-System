# Use an official Python base image with minimal footprint
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install minimal system dependencies required for OpenCV and Flask
RUN apt-get clean && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker caching
COPY requirements.txt .

# Install Python dependencies in smaller batches to avoid memory issues
RUN pip install --upgrade pip
RUN pip install opencv-python==4.8.1.78 numpy==1.24.3
RUN pip install pika==1.3.2 Flask==3.1.1 waitress==3.0.2

# Copy rest of the code
COPY . .

# Expose port for Flask app (Waitress)
EXPOSE 8000

# Default command to run the Flask + Waitress server
CMD ["python", "stream_api.py"]

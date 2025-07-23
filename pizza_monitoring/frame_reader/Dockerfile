FROM python:3.9-slim

# Install minimal system dependencies
RUN apt-get clean && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory inside container
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/

# Install Python dependencies in smaller batches
RUN pip install --upgrade pip
RUN pip install opencv-python==4.8.1.78
RUN pip install numpy==1.24.3 pika==1.3.2

# Copy project code
COPY . /app

# Run the reader
CMD ["python", "reader.py"]

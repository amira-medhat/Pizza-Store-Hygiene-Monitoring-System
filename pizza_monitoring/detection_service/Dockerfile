FROM python:3.9-slim

# Install required system libraries for OpenCV one by one
RUN apt-get update
RUN apt-get install -y --no-install-recommends libgl1-mesa-glx
RUN apt-get install -y --no-install-recommends libglib2.0-0
RUN apt-get install -y --no-install-recommends libsm6
RUN apt-get install -y --no-install-recommends libxext6
RUN apt-get install -y --no-install-recommends libxrender-dev
RUN rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only what's needed for pip install
COPY requirements.txt .

# Install minimal dependencies one by one to avoid I/O errors
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir opencv-python-headless==4.8.1.78
RUN pip install --no-cache-dir numpy==1.24.3
RUN pip install --no-cache-dir pika==1.3.2
RUN pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir ultralytics==8.3.166

# Copy project code
COPY . .

# Create directory for weights
RUN mkdir -p /app/weights

# Create shared directory and copy config.py
RUN mkdir -p /app/shared
COPY shared/config.py /app/shared/

# Initialize database and start detector
CMD ["python", "detector.py"]

FROM python:3.9-slim

# Install required system libraries for OpenCV one by one
RUN apt-get update
RUN apt-get install -y --no-install-recommends libgl1-mesa-glx
RUN rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies one by one
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir opencv-python-headless==4.8.1.78
RUN pip install --no-cache-dir numpy==1.24.3
RUN pip install --no-cache-dir pika==1.3.2

# Copy project code
COPY . .

# Create shared directory and copy config.py
RUN mkdir -p /app/shared
COPY shared/config.py /app/shared/

# Run the reader
CMD ["python", "reader.py"]
